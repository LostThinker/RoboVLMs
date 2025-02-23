#!/bin/bash

REPO_URL="https://github.com/LostThinker/RoboVLMs.git"

# 克隆 Git 仓库
echo "Cloning the repository..."
git clone $REPO_URL


conda create -n robovlms python=3.10 -y

conda activate robovlms
conda install cudatoolkit cudatoolkit-dev -y

cd ./RoboVLMs
pip install -e .

pip install transformers==4.44.0

git lfs install
cd .vlms & git clone https://huggingface.co/Alpha-VLA/Paligemma



echo "Start co-finetuning"
bash scripts/run.sh configs/oxe/configs/calvin_finetune/finetune_paligemma_cont-lstm-post_full-ft_text_vision_wd=0_ws-8_act-10.json


