from dataclasses import dataclass


@dataclass
class Config:
    video_path: str
    result_dir: str
    # path to SAM checkpoint
    sam_checkpoint: str
    # path to SAM2 checkpoint
    sam2_checkpoint: str
    # Name of SAM2 config
    sam2_config: str
    # model to run
    runner: str = "final"
    # number of prompt points to image SAM to obtain first frame masks
    start_points_per_side: int = 16
    # number of prompt points to image SAM to obtain next frame masks
    prompt_points_per_side: int = 8
    # number of frames we keep in SAM2 memory while propagating
    memory_size: int = 10
    # fps of result video
    fps: int = 8
    # resolution of result video
    resolution: int = 1
    # number of frames to run (by default the whole video)
    num_iterations: int = -1
    # debug mode (Saves intermediate results)
    debug: bool = False
    # Loads segmentations from backwards or forwards if exists
    resume: bool = True
    # Root directory for prompts (must contain: masks/, optional: diffs/)
    prompt_dir: str = ""
    # Prompt filter threshold
    filter_threshold: float = 0.08
    # threshold for prompt masks refinment
    lambda_ref: float = 0.7
    # threshold for considering mask in filtration step
    lambda_val: float = 0.7
    # threshold for mask temporal merging
    lambda_merge: float = 0.9


@dataclass
class MasksConverterConfig:
    source_path: str
    output_path: str
