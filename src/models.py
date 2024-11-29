from config import Config

from sam2.build_sam import build_sam2_video_predictor
from sam2.utils.amg import build_point_grid
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy import ndimage
from utils import (
    save_segmentations,
    mask_images_parallel,
    make_video,
    load_segmentations,
    find_intersections,
    enable_tf32_and_cudnn_for_ampere,
    sample_points_outside_mask,
    safe_state,
    segmentations_to_binary,
    save_binary_masks,
)

import json
import time
from natsort import natsorted
import cv2
import random
from collections import defaultdict


class BaseRunner:

    def __init__(self, cfg: Config, exp_name=None):
        enable_tf32_and_cudnn_for_ampere()

        self.start_time = time.time()
        self.cfg = cfg

        self.video_path = cfg.video_path
        self.device = "cuda"

        self.sam1 = sam_model_registry["vit_h"](checkpoint=cfg.sam_checkpoint).to(self.device)

        self.predictor = build_sam2_video_predictor(
            os.path.join("configs/sam2/", cfg.sam2_config), cfg.sam2_checkpoint, device=self.device
        )

        self.predictor.max_cond_frames_in_attn = cfg.memory_size

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            self.inference_state = self.predictor.init_state(video_path=self.video_path)

        self.num_iterations = min(self.cfg.num_iterations, self.inference_state["num_frames"] - 1)
        if self.num_iterations < 0:
            self.num_iterations = self.inference_state["num_frames"] - 1

        if not exp_name:
            if cfg.result_dir.endswith("/"):
                exp_name = self.cfg.result_dir.split("/")[-2]
            else:
                exp_name = cfg.result_dir.split("/")[-1]

        if self.cfg.debug:
            self.debug_dir = os.path.join(self.cfg.result_dir, "debug")

        self.exp_name = exp_name
        if self.cfg.prompt_dir:

            self.prompt_masks_dir = os.path.join(self.cfg.prompt_dir, "masks")
            self.prompt_diffs_dir = os.path.join(self.cfg.prompt_dir, "diffs")

            self.prompt_names = natsorted(
                [
                    p
                    for p in os.listdir(self.prompt_masks_dir)
                    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png"]
                ]
            )

        self.init_segmentations()

        os.makedirs(cfg.result_dir, exist_ok=True)

    def read_prompt_mask(self, idx):
        mask_path = os.path.join(self.prompt_masks_dir, self.prompt_names[idx])
        mask = np.array(
            Image.open(mask_path).resize(
                (self.inference_state["video_width"], self.inference_state["video_height"])
            )
        ).astype("uint8")

        return mask

    def read_prompt_diff(self, idx):
        diff_path = os.path.join(self.prompt_diffs_dir, self.prompt_names[idx])
        diff = (
            np.array(
                Image.open(diff_path).resize(
                    (self.inference_state["video_width"], self.inference_state["video_height"])
                )
            ).astype("float")
            / 255
        )
        return diff

    def add_masks_to_predictor(self, frame_idx, masks):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            dtype = next(self.predictor.parameters()).dtype
            for mask_idx, mask in masks.items():
                mask_tensor = torch.tensor(mask, dtype=dtype, device=self.device)

                _, out_obj_ids, out_mask_logits = self.predictor.add_new_mask(
                    inference_state=self.inference_state,
                    frame_idx=frame_idx,
                    obj_id=mask_idx,
                    mask=mask_tensor,
                )

    def generate_masks(self, frame_idx, points_per_side=None, point_grids=None):
        """Single frame mask generation."""

        mask_generator = SamAutomaticMaskGenerator(
            self.sam1,
            points_per_side=points_per_side,
            point_grids=point_grids,
        )

        frame_path = os.path.join(
            self.video_path, natsorted(os.listdir(self.video_path))[frame_idx]
        )
        frame = np.array(Image.open(frame_path).convert("RGB"))
        masks = mask_generator.generate(frame)

        return {idx: m["segmentation"] for idx, m in enumerate(masks)}

    def init_segmentations(self):
        # look for first frame with prompt masks
        for start_frame_idx in range(self.num_iterations):
            if self.cfg.prompt_dir:
                masks = self.propose_masks(start_frame_idx)
            else:
                masks = self.generate_masks(
                    start_frame_idx, points_per_side=self.cfg.start_points_per_side
                )

            if masks:
                break

        print("Starting from frame", start_frame_idx)
        self.start_frame_idx = start_frame_idx
        self.segmentations = {start_frame_idx: masks}
        self.labels_cnt = len(list(self.segmentations[start_frame_idx]))
        self.update_propagator = True

    def get_prompt_points(self, frame_idx):
        """Sample points inside prompt points regions."""

        def determine_points(size):
            return max(10, int(0.01 * size))

        prompt_mask = self.read_prompt_mask(frame_idx)

        num_labels, labels = cv2.connectedComponents(prompt_mask)
        sampled_points = []

        for label in range(1, num_labels):
            object_mask = (labels == label).astype(np.uint8)

            object_size = np.sum(object_mask)

            num_points = determine_points(object_size)

            coordinates = np.argwhere(object_mask > 0)

            normalized_coords = coordinates.astype("float")
            normalized_coords[:, 0] = coordinates[:, 0] / prompt_mask.shape[0]
            normalized_coords[:, 1] = coordinates[:, 1] / prompt_mask.shape[1]

            if len(coordinates) >= num_points:
                sampled_indices = random.sample(range(len(coordinates)), num_points)
                object_points = normalized_coords[sampled_indices]
                sampled_points.extend(object_points.tolist())

        coordinates_list = [np.array([coord[1], coord[0]]) for coord in sampled_points]

        return np.array(coordinates_list)

    def propose_masks(self, frame_idx, propagated_masks=None):

        def calculate_iom(prompt, mask):
            """Calculate Intersection over mask"""
            intersection = np.logical_and(prompt > 0, mask > 0).sum()
            return intersection / (mask > 0).sum()

        if self.cfg.prompt_dir:
            points = self.get_prompt_points(frame_idx)
            if not len(points):
                return {}
        else:
            points = build_point_grid(self.cfg.prompt_points_per_side)

        if propagated_masks is not None:
            binary_mask = np.zeros(
                (self.inference_state["video_height"], self.inference_state["video_width"])
            )
            for _, mask in propagated_masks.items():
                binary_mask += mask

            dilated_mask = ndimage.binary_dilation(binary_mask.squeeze(), iterations=10)
            points = sample_points_outside_mask(points, dilated_mask)

        if not len(points):
            return {}

        prompt_mask = self.read_prompt_mask(frame_idx)
        masks = self.generate_masks(frame_idx, point_grids=[points])

        if self.cfg.prompt_dir:
            prompt_mask = self.read_prompt_mask(frame_idx)
            filtered_masks = {}
            for idx, mask in enumerate(masks.values()):
                if calculate_iom(prompt_mask, mask) > self.cfg.lambda_ref:
                    filtered_masks[idx] = mask

            return filtered_masks
        else:
            return masks

    def reset_state(self):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            self.predictor.reset_state(self.inference_state)

    def save(self, save_dir):
        segmentations_dir = os.path.join(save_dir, "segmentations")
        save_segmentations(self.segmentations, segmentations_dir)

        masked_images = mask_images_parallel(
            self.inference_state["num_frames"],
            self.segmentations,
            self.video_path,
            self.cfg.resolution,
            n_labels=self.labels_cnt,
        )

        make_video(masked_images, save_dir, self.exp_name, self.cfg.fps)

    def make_report(self):
        end_time = time.time()
        elapsed_time = end_time - self.start_time

        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        formatted_time = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

        report = {
            "num_labels": self.labels_cnt,
            "running_time": formatted_time,
        }

        json.dump(report, open(os.path.join(self.cfg.result_dir, "report.json"), "w"))


class DualRunner(BaseRunner):

    def handle_intersections(self, intersecting_labels):
        """Joins intersecting labels through all segmentations."""

        for frame_idx, masks in self.segmentations.items():
            for target_label, source_labels in intersecting_labels.items():
                for source_label in source_labels:
                    if source_label not in masks:
                        continue

                    if target_label not in masks:
                        self.segmentations[frame_idx][target_label] = np.zeros(
                            (
                                self.inference_state["video_height"],
                                self.inference_state["video_width"],
                            ),
                            dtype=bool,
                        )

                    self.segmentations[frame_idx][target_label] += masks[source_label].astype(bool)
                    self.segmentations[frame_idx].pop(source_label)

    def reformat_labels(self):
        """Make labels continuous."""

        unique_labels = set()
        for frame_data in self.segmentations.values():
            unique_labels.update(frame_data.keys())

        self.labels_cnt = len(unique_labels)

        # Create a mapping from original labels to continuous labels
        label_mapping = {label: new_label for new_label, label in enumerate(sorted(unique_labels))}

        # Apply the mapping to reformat the labels in segmentation masks
        reformatted_masks = {}
        for frame_idx, frame_data in self.segmentations.items():
            reformatted_frame_data = {
                label_mapping[label]: mask for label, mask in frame_data.items()
            }
            reformatted_masks[frame_idx] = reformatted_frame_data

        self.segmentations = reformatted_masks

    def add_segmentations(self, frame_idx, new_segmentations):
        if frame_idx not in self.segmentations:
            self.segmentations[frame_idx] = {}

        for _, mask in new_segmentations.items():
            self.segmentations[frame_idx][self.labels_cnt] = mask
            self.labels_cnt += 1

    def init_forward_propagator(self):
        self.reset_state()

        free_mem_size = self.cfg.memory_size
        cur_frame_idx = self.start_frame_idx + len(self.segmentations)

        for frame_idx in range(cur_frame_idx - 1, self.start_frame_idx - 1, -1):
            if self.segmentations[frame_idx] and free_mem_size > 0:
                self.add_masks_to_predictor(frame_idx, self.segmentations[frame_idx])
                free_mem_size -= 1

            if free_mem_size == 0:
                break

        self.forward_propagator = self.predictor.propagate_in_video(
            self.inference_state, start_frame_idx=cur_frame_idx
        )
        self.update_propagator = False

    def propagate_forward(self, target_frame_idx, skip_empty=False):
        """Propagate forward until target_frame_idx is reached."""

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            while True:
                out_frame_idx, out_obj_ids, out_mask_logits = next(self.forward_propagator)
                target_frame_propagated = {}
                if out_frame_idx == target_frame_idx:
                    for i, out_obj_id in enumerate(out_obj_ids):
                        mask = (out_mask_logits[i] > 0.0).detach().cpu().numpy().squeeze()
                        if skip_empty and mask.sum() == 0:
                            continue

                        target_frame_propagated[out_obj_id] = mask

                    return {target_frame_idx: target_frame_propagated}

    def save_debug_forward(self, propagated, proposals):
        proposals_dir = os.path.join(self.debug_dir, "forward_proposals")
        propagated_dir = os.path.join(self.debug_dir, "forward_propagated")

        save_segmentations(propagated, propagated_dir)
        save_segmentations(proposals, proposals_dir)

    def run_forward(self):
        with open(f"{self.cfg.result_dir}/cfg.json", "w") as f:
            json.dump(vars(self.cfg), f)

        for left_frame_idx in tqdm(
            range(self.start_frame_idx, self.num_iterations), desc="Main progress"
        ):
            right_frame_idx = left_frame_idx + 1

            if self.update_propagator:
                self.init_forward_propagator()

            right_frame_propagated = self.propagate_forward(right_frame_idx, skip_empty=True)
            self.segmentations.update(right_frame_propagated)

            intersecting_labels = find_intersections(
                right_frame_propagated[right_frame_idx], self.cfg.lambda_merge
            )
            if intersecting_labels:
                self.update_propagator = True
                self.handle_intersections(intersecting_labels)

            right_frame_proposals = self.propose_masks(
                right_frame_idx, right_frame_propagated[right_frame_idx]
            )

            if self.cfg.debug:
                self.save_debug_forward(
                    right_frame_propagated, {right_frame_idx: right_frame_proposals}
                )

            if right_frame_proposals:
                self.add_segmentations(right_frame_idx, right_frame_proposals)
                self.update_propagator = True

    def init_backward_propagator(self, cur_frame_idx):
        self.reset_state()

        free_mem_size = self.cfg.memory_size
        for frame_idx in range(cur_frame_idx + 1, self.num_iterations + 1):
            if self.segmentations[frame_idx] and free_mem_size > 0:
                self.add_masks_to_predictor(frame_idx, self.segmentations[frame_idx])
                free_mem_size -= 1

            if free_mem_size == 0:
                break

        if free_mem_size == self.cfg.memory_size:
            return False

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            self.backward_propagator = self.predictor.propagate_in_video(
                self.inference_state, reverse=True
            )
        self.update_propagator = False

        return True

    def propagate_backward(self, target_frame_idx, skip_empty=False):
        """Propagate backward until target_frame_idx is reached."""

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            while True:
                out_frame_idx, out_obj_ids, out_mask_logits = next(self.backward_propagator)
                target_frame_propagated = {}
                if out_frame_idx == target_frame_idx:
                    for i, out_obj_id in enumerate(out_obj_ids):
                        mask = (out_mask_logits[i] > 0.0).detach().cpu().numpy().squeeze()
                        if skip_empty and mask.sum() == 0:
                            continue

                        target_frame_propagated[out_obj_id] = mask

                    return {target_frame_idx: target_frame_propagated}

    def save_debug_backward(self, backward):
        backward_dir = os.path.join(self.debug_dir, "backward_propagated")
        save_segmentations(backward, backward_dir)

    def run_backward(self):
        self.update_propagator = True

        for right_frame_idx in tqdm(range(self.num_iterations, 0, -1), desc="Backward progress"):
            left_frame_idx = right_frame_idx - 1

            if self.update_propagator:
                if not self.init_backward_propagator(left_frame_idx):
                    print(f"{left_frame_idx}: No prompts found. Continue")
                    continue

            left_frame_propagated = self.propagate_backward(left_frame_idx, skip_empty=True)

            if self.cfg.debug:
                self.save_debug_backward(left_frame_propagated)

            if left_frame_idx not in self.segmentations:
                self.segmentations[left_frame_idx] = {}

            if set(self.segmentations[left_frame_idx]) - set(left_frame_propagated[left_frame_idx]):
                self.update_propagator = True

            for label, mask in left_frame_propagated[left_frame_idx].items():
                if label in self.segmentations[left_frame_idx]:
                    self.segmentations[left_frame_idx][label] += mask
                else:
                    self.segmentations[left_frame_idx][label] = mask

            intersecting_labels = find_intersections(
                self.segmentations[left_frame_idx], self.cfg.lambda_merge
            )

            if intersecting_labels:
                self.update_propagator = True
                self.handle_intersections(intersecting_labels)

    def run(self):
        backward_path = os.path.join(self.cfg.result_dir, "backward")
        forward_path = os.path.join(self.cfg.result_dir, "forward")

        if self.cfg.resume:
            if os.path.exists(backward_path):
                self.segmentations = load_segmentations(
                    os.path.join(backward_path, "segmentations")
                )
                return
            elif os.path.exists(forward_path):
                self.segmentations = load_segmentations(os.path.join(forward_path, "segmentations"))
                self.run_backward()
                self.reformat_labels()
                self.save(backward_path)
                return

        self.run_forward()
        self.reformat_labels()
        self.save(forward_path)

        self.run_backward()
        self.reformat_labels()
        self.save(backward_path)


class FinalRunner(DualRunner):

    def pred_masks(self, target_frame_idx):
        self.reset_state()

        free_mem_size = self.cfg.memory_size
        frames = range(0, self.num_iterations + 1)
        sorted_frames = sorted(frames, key=lambda x: abs(x - target_frame_idx))

        for frame_idx in sorted_frames:
            if frame_idx == target_frame_idx:
                continue

            if self.segmentations[frame_idx]:
                self.add_masks_to_predictor(frame_idx, self.segmentations[frame_idx])
                free_mem_size -= 1

            if free_mem_size == 0:
                break

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            final_propagator = self.predictor.propagate_in_video(
                self.inference_state, start_frame_idx=target_frame_idx
            )

            _, out_obj_ids, out_mask_logits = next(final_propagator)
            target_frame_propagated = {}
            for i, out_obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits[i] > 0.0).detach().cpu().numpy().squeeze()
                if mask.sum() == 0:
                    continue

                target_frame_propagated[out_obj_id] = mask

        return target_frame_propagated

    def save_debug_final(self, final):
        final_dir = os.path.join(self.debug_dir, "final")
        save_segmentations(final, final_dir)

    def run_final(self):
        for left_frame_idx in tqdm(range(self.num_iterations + 1), desc="Final progress"):
            masks = self.pred_masks(left_frame_idx)
            if not masks:
                continue

            if self.cfg.debug:
                self.save_debug_final({left_frame_idx: masks})

            if left_frame_idx not in self.segmentations:
                self.segmentations[left_frame_idx] = {}

            for label, mask in masks.items():
                if label in self.segmentations[left_frame_idx]:
                    self.segmentations[left_frame_idx][label] += mask
                else:
                    self.segmentations[left_frame_idx][label] = mask

            intersecting_labels = find_intersections(
                self.segmentations[left_frame_idx], self.cfg.lambda_merge
            )

            if intersecting_labels:
                self.update_propagator = True
                self.handle_intersections(intersecting_labels)

    def cnt_stability_ratio_by_masks_diffs(self):
        def calculate_iom(prompt, mask, mask_size):

            intersection = np.logical_and(prompt > 0, mask > 0).sum()
            return intersection / mask_size

        segmentations_cnt = defaultdict(list)
        mask_sizes = {}

        for _, masks in self.segmentations.items():
            for label, mask in masks.items():
                mask_sizes[label] = max(mask_sizes.get(label, 0), mask.sum())

        for frame_idx, masks in self.segmentations.items():
            diff_prompt = self.read_prompt_diff(frame_idx)
            mask_prompt = self.read_prompt_mask(frame_idx)

            for label, mask in masks.items():
                if mask.sum() == 0:
                    continue

                if calculate_iom(mask_prompt, mask, mask.sum()) > self.cfg.lambda_val:
                    segmentations_cnt[label].append(
                        diff_prompt[mask].mean()
                        * calculate_iom(mask_prompt, mask, mask_sizes[label])
                    )

        metrics = {}
        for label, vals in segmentations_cnt.items():
            metrics[label] = np.array(vals).mean()

        return metrics

    def _filter(self, statbility_ratio):
        popular_labels = [
            label for label, ratio in statbility_ratio.items() if ratio > self.cfg.filter_threshold
        ]

        self.segmentations = {
            frame_idx: {label: mask for label, mask in masks.items() if label in popular_labels}
            for frame_idx, masks in self.segmentations.items()
        }

    def run(self):
        super().run()
        final_path = os.path.join(self.cfg.result_dir, "final")

        if self.cfg.resume and os.path.exists(final_path):
            self.segmentations = load_segmentations(os.path.join(final_path, "segmentations"))
        else:
            self.run_final()
            self.reformat_labels()
            self.save(final_path)

        if self.cfg.prompt_dir:
            filter_path = os.path.join(
                self.cfg.result_dir,
                f"final_filtered_{self.cfg.filter_threshold}_{self.cfg.lambda_val}",
            )
            if self.cfg.resume and os.path.exists(filter_path):
                self.segmentations = load_segmentations(os.path.join(filter_path, "segmentations"))
            else:
                stability_ratio = self.cnt_stability_ratio_by_masks_diffs()
                self._filter(stability_ratio)
                self.save(filter_path)

        if self.cfg.prompt_dir:
            bin_masks_output = os.path.join(self.cfg.result_dir, "binary_masks")
            bin_masks = segmentations_to_binary(self.segmentations)
            save_binary_masks(bin_masks, bin_masks_output)

class RunnerFactory:
    runners = {
        "dual": DualRunner,
        "final": FinalRunner,
    }

    @classmethod
    def start(cls, cfg: Config) -> BaseRunner:
        assert cfg.runner in cls.runners, f"Invalid runner type: {cfg.runner}"

        safe_state()

        runner_class = cls.runners[cfg.runner]
        return runner_class(cfg)
