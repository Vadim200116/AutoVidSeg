# Automatic Video Segmentation Using SAM2
## [Project Page](https://transient-3dgs.github.io/) | [Paper](https://arxiv.org/abs/2412.00155)

## Abstract
This project presents a video segmentation tool based on SAM2, offering two operational modes:
- **Promptable Mode:** Refines existing masks using inputs such as L1 distances and predicted masks from [T-3DGS](https://github.com/Vadim200116/T-3DGS).
- **Promptless Mode:** Performs automatic segmentation without prior masks, leveraging the capabilities of SAM1 and SAM2.

Our framework facilitates efficient and accurate video segmentation, essential for tasks like mask refinement and object tracking in videos.

## Installation
1. **Clone the Repository:**

```bash
git clone https://github.com/yourusername/Automatic-Video-Segmentation-SAM2.git 
cd Automatic-Video-Segmentation-SAM2
```

2. **Install Requirements:**
Use pip to install the necessary packages:
```bash
pip install -r requirements.txt
```
3. **Download SAM Checkpoints:**
Download SAM1 and SAM2 checkpoints. You can refer to original repositories for this precedure.

## Directory Structure
The tool processes sequential images in SAM2 format:
```bash
|-<video dir>/
    |-000001.jpg
    |-000002.jpg
    |-000003.jpg
    |-000004.jpg
    ...
```

## Usage
### Promptable mode

Designed for refining existing masks, as described in [our paper](https://arxiv.org/abs/2412.00155). Please refer to our [companion repository](https://github.com/Vadim200116/T-3DGS) to obtain prompt.
#### Input Requirements:

- **Images:** Sequential images in SAM2 format.
- **L1 Distances:** Between rendered and ground truth frames.
- **Predicted Transient Masks:** Initial masks to be refined.

#### Directory Structure for Prompts:
Organize your prompt directory as follows:
```bash
|-<prompt dir>/
    |-masks/
        |-000001.png
        |-000002.png
        |-000003.png
        |-000004.png
        ...
    |-diffs/
        |-000001.png
        |-000002.png
        |-000003.png
        |-000004.png
        ...
```

#### Running Promptable Mode

To run the segmentation in promptable mode:
```bash
python3 src/video_segmentor.py \
  --video_path /path/to/images \
  --result_dir /path/to/save/results \
  --sam_checkpoint /path/to/sam1_checkpoint \
  --sam2_checkpoint /path/to/sam2_checkpoint \
  --sam2_config config_name \
  --prompt_dir /path/to/prompt_directory
```
For additional arguments and configurations, please refer to `src/config.py`.

#### Benchmarking in Promptable Mode
To run benchmarking in promptable mode:
```bash
bash examples/tmr_benchmark.sh
```

### Promptless Mode
Performs automatic segmentation without any prior masks.

#### How It Works
- **Segmentation with SAM1:** Segments each image individually.
- **Propagation with SAM2:** Propagates and merges segmentations across frames for temporal consistency.

#### Running Promptless Mode
To run the segmentation in promptless mode:
```bash
python3 src/video_segmentor.py \
  --video_path /path/to/images \
  --result_dir /path/to/save/results \
  --sam_checkpoint /path/to/sam1_checkpoint \
  --sam2_checkpoint /path/to/sam2_checkpoint \
  --sam2_config config_name
```

## Citation
If you find this work useful in your research, please consider citing:
```bibtex
@misc{pryadilshchikov2024t3dgsremovingtransientobjects,
      title={T-3DGS: Removing Transient Objects for 3D Scene Reconstruction}, 
      author={Vadim Pryadilshchikov and Alexander Markin and Artem Komarichev and Ruslan Rakhimov and Peter Wonka and Evgeny Burnaev},
      year={2024},
      eprint={2412.00155},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.00155}, 
}
```