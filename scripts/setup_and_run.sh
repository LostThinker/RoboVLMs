#!/bin/bash

REPO_URL = "https://github.com/LostThinker/RoboVLMs.git"

# 克隆 Git 仓库
echo "Cloning the repository..."
git clone -b co_finetune $REPO_URL

echo "Create conda env"
conda create -n robovlms python=3.10 -y

conda activate robovlms
conda install cudatoolkit cudatoolkit-dev -y

cd ./RoboVLMs

pip install -e .
pip install transformers==4.49.0

echo "download dataset"
mkdir -p datasets/open-x-embodiment
cd datasets/open-x-embodiment
wget -r -nH --cut-dirs=4 --reject="index.html*" https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/
mv bridge_dataset bridge_orig

cd ../..

echo "Models need to be downloaded manually"
git lfs install
cd .vlms & git clone https://huggingface.co/Alpha-VLA/Paligemma
cd ..


echo "Start training"
bash scripts/run.sh configs/oxe_training/finetune_paligenmma_cont-lstm-post_full-ft_text_vision_wd-0_use-hand_ws-16_act-10_bridge_finetune.json


