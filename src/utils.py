import os
import numpy as np
import concurrent.futures
from PIL import Image
import imageio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm
from numba import njit
import torch
import random
from natsort import natsorted


def enable_tf32_and_cudnn_for_ampere():
    """From SAM2 demo noutbook."""

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


@njit
def calculate_iou(mask1, mask2):
    intersection = np.sum(mask1 & mask2)
    union = np.sum(mask1 | mask2)
    return intersection / union


def save_segmentations(segmentations, results_dir):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    for frame_ix, segmentation in segmentations.items():
        frame_dir = os.path.join(results_dir, f"{frame_ix}")
        os.makedirs(frame_dir, exist_ok=True)

        for label, mask in segmentation.items():
            mask_file = os.path.join(frame_dir, f"{label}.npy")
            np.save(mask_file, mask)


def load_segmentations(segmentations_dir, target_size=None):
    segmentations = {}

    for frame_dir_name in os.listdir(segmentations_dir):
        frame_dir_path = os.path.join(segmentations_dir, frame_dir_name)

        if os.path.isdir(frame_dir_path):
            frame_ix = int(frame_dir_name)
            segmentations[frame_ix] = {}

            # Load each numpy binary mask
            for mask_file_name in os.listdir(frame_dir_path):
                label = int(mask_file_name.replace(".npy", ""))
                mask_file_path = os.path.join(frame_dir_path, mask_file_name)
                mask = np.load(mask_file_path)
                if target_size is not None and mask.shape != target_size:
                    mask_img = Image.fromarray(mask)
                    mask = np.array(mask_img.resize(target_size))

                segmentations[frame_ix][label] = mask

    return segmentations


def make_video(frames, result_dir, gif_name, fps):
    os.makedirs(result_dir, exist_ok=True)
    writer = imageio.get_writer(f"{result_dir}/{gif_name}.mp4", fps=fps)
    for canvas in frames:
        writer.append_data(np.array(canvas))
    writer.close()


def mask_img(mask, obj_id=None, random_color=False, n_labels=100):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        tab20c = plt.get_cmap("tab20c")
        colors = tab20c(np.linspace(0, 1, n_labels))
        cmap = ListedColormap(colors)
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)  # Shape (h, w, 4)
    return mask_image


def mask_pil_image(segments, video_path, frame_idx, labels=None, resolution=1, n_labels=100):
    vis_frame_path = os.path.join(video_path, natsorted(os.listdir(video_path))[frame_idx])
    original_image = Image.open(vis_frame_path).convert("RGBA")
    if resolution > 1:
        original_image = original_image.resize(
            (original_image.width // resolution, original_image.height // resolution),
            resample=Image.BILINEAR,
        )

    original_array = np.array(original_image)

    blended_image = original_array

    for out_obj_id, out_mask in segments.get(frame_idx, {}).items():
        if labels is not None:
            if out_obj_id not in labels:
                continue
        mask_image = mask_img(out_mask, obj_id=out_obj_id, n_labels=n_labels)
        mask_image = (mask_image * 255).astype(np.uint8)

        mask_pil = Image.fromarray(mask_image)
        mask_pil = mask_pil.resize(
            (original_image.width, original_image.height), resample=Image.BILINEAR
        )
        mask_array = np.array(mask_pil)

        blended_image = np.where(mask_array[:, :, 3:] > 0, mask_array, blended_image)

    return Image.fromarray(blended_image)


def mask_images_parallel(num_frames, masks, video_path, resolution=1, n_labels=100):
    masked_images_dict = {}

    def process_frame(frame_idx):
        return mask_pil_image(
            masks, video_path, frame_idx, resolution=resolution, n_labels=n_labels
        )

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Creating a dictionary of future to frame index
        futures = {
            executor.submit(process_frame, frame_idx): frame_idx for frame_idx in range(num_frames)
        }

        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures), desc="Masking images"
        ):
            frame_idx = futures[future]
            try:
                masked_img = future.result()
                masked_images_dict[frame_idx] = masked_img
            except Exception as e:
                print(f"Frame {frame_idx} generated an exception: {e}")

    masked_images = []
    for k in sorted(list(masked_images_dict)):
        masked_images.append(masked_images_dict[k])

    return masked_images


def find_intersections(masks, iou_threshold):
    """Returns dict where value is set of intersecting labels and key is their new label."""

    labels = list(masks.keys())
    intersections = {}

    def process_pair(pair):
        label1, label2 = pair
        mask1 = masks[label1]
        mask2 = masks[label2]

        iou = calculate_iou(mask1, mask2)
        return (label1, label2) if iou > iou_threshold else None

    label_pairs = [
        (labels[i], labels[j]) for i in range(len(labels)) for j in range(i + 1, len(labels))
    ]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = filter(None, executor.map(process_pair, label_pairs))

    for label1, label2 in results:
        intersections.setdefault(label1, set()).add(label2)
        intersections.setdefault(label2, set()).add(label1)

    # Merging intersecting labels groups
    intersecting_labels = {}
    visited = set()

    def dfs(label, group):
        stack = [label]
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                group.add(current)
                stack.extend(intersections.get(current, []))

    for label in labels:
        if label not in visited:
            group = set()
            dfs(label, group)
            if len(group) > 1:
                lowest_label = min(group)
                group.remove(lowest_label)
                # Merge all intersecting masks in lowest label
                intersecting_labels[lowest_label] = group

    return intersecting_labels


def sample_points_outside_mask(points, binary_mask):
    # Scale the points to match the binary mask dimensions
    mask_shape = binary_mask.shape
    points_scaled = (points * [mask_shape[1], mask_shape[0]]).astype(int)

    # Filter points outside the binary mask
    mask_flat = binary_mask[points_scaled[:, 1], points_scaled[:, 0]]
    points_outside_mask = points[~mask_flat]

    return points_outside_mask


def safe_state():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


def segmentations_to_binary(segmentations):
    bin_masks = {}
    for frame_idx, masks in tqdm(segmentations.items()):
        bin_mask = None
        for mask in masks.values():
            if bin_mask is None:
                bin_mask = mask
            else:
                bin_mask += mask

        if bin_mask is not None:
            bin_masks[frame_idx] = bin_mask

    return bin_masks


def save_binary_masks(bin_masks, output_path):
    os.makedirs(output_path, exist_ok=True)
    for frame_idx, mask in tqdm(bin_masks.items()):
        Image.fromarray(~mask).save(os.path.join(output_path, f"{frame_idx}.png"))
