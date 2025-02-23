#!/usr/bin/env bash

subfix=`date "+%H-%M"`

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=6


python main.py \
  --exp_name ${subfix} \
  ${@:1} \
  --gpus 1