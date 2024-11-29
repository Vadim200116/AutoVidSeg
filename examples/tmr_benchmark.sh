#!/bin/bash

DATA_DIR=/data
DATASET_NAME=t-3dgs
MODEL_DIR=/output/t-3dgs
SEGMENTATIONS_DIR=/output/tmr

for SCENE in lab1 lab2 library anti_stress office
do
    python3 src/video_segmentor.py \
        --result_dir="${SEGMENTATIONS_DIR}/${DATASET_NAME}/${SCENE}" \
        --video_path="${DATA_DIR}/${DATASET_NAME}/${SCENE}/images_tmr" \
        --prompt_dir="${MODEL_DIR}/${DATASET_NAME}/${SCENE}/prompt" \
        --sam_checkpoint="/checkpoints/sam_vit_h_4b8939.pth" \
        --sam2_checkpoint="/checkpoints/sam2_hiera_large.pt" \
        --sam2_config="sam2_hiera_l.yaml"
done