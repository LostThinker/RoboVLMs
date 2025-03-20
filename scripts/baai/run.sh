#!/usr/bin/env bash

source /share/project/lxh/conda/bin/activate robovlm
wandb login 81ad29ffb0afe40c1122f47e6b1b9698f8deea77

CUR_DIR=$(cd $(dirname $0); pwd)
chmod 777 -R $CUR_DIR

WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-23456}

#LEGACY DDP CONFIGS
ports=(`echo $MASTER_PORT | tr ',' ' '`)
port=${ports[0]}

GPUS_PER_NODE=$(nvidia-smi --query-gpu=count --format=csv,noheader -i 0)

echo "total workers: ${PET_WORLD_SIZE}"
echo "cur worker id: ${PET_NODE_RANK}"
echo "gpus per worker: ${GPUS_PER_NODE}"
echo "master ip: ${MASTER_ADDR}"
echo "master port: ${port}"

set -x
export PYTHONUNBUFFERED=1
# export NCCL_DEBUG=INFO
# export NCCL_NET_GDR_LEVEL=0
# export NCCL_IB_DISABLE=1
export NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_BLOCKING_WAIT=1
# export CUDA_LAUNCH_BLOCKING=1
# export FSDP_CPU_RAM_EFFICIENT_LOADING=1
export OMP_NUM_THREADS=16
export NCCL_IB_GID_INDEX=7
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_MIN_NCHANNELS=32
export NCCL_IB_QPS_PER_CONNECTION=4

# export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1200  
export TORCH_NCCL_ENABLE_MONITORING=0
## huggingface settings if needed
# export HF_HOME=""
export HF_ENDPOINT='https://hf-mirror.com'


# convert deepspeed checkpoint first
if [ $NODE_ID == "0" ]; then
  echo "---------- Converting deepspeed checkpoint to fp32. ----------"
  python3 tools/convert_deepspeed_to_fp32.py ${@:1}
fi

subfix=`date "+%H-%M"`

echo "RUNNING:"
echo torchrun \
    --nnodes $PET_WORLD_SIZE \
    --node_rank $RANK \
    --nproc_per_node $GPUS_PER_NODE \
    --master_addr $MASTER_ADDR \
    --master_port $port \
    main.py \
    --exp_name ${subfix} \
    ${@:1} \
    --gpus $GPUS_PER_NODE \
    --num_nodes $WORLD_SIZE

torchrun \
    --nnodes $PET_WORLD_SIZE \
    --node_rank $RANK \
    --nproc_per_node $GPUS_PER_NODE \
    --master_addr $MASTER_ADDR \
    --master_port $port \
    main.py \
    --exp_name ${subfix} \
    ${@:1} \
    --gpus $GPUS_PER_NODE \
    --num_nodes $WORLD_SIZE