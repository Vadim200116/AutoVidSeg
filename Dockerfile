FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENV DEBIAN_FRONTEND=noninteractive

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"

WORKDIR /app

RUN apt-get update && \
    apt-get -y install build-essential git libgl1-mesa-glx libglib2.0-0

COPY requirements.txt .
COPY src/ ./src/

RUN pip install -r requirements.txt
