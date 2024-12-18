#!/usr/bin/env bash

T=`date +%m%d%H%M` # 日期

# -------------------------------------------------- #
# Usually you only need to customize these variables #
CFG=$1   # 第一个参数为cfg路径                          #
GPUS=$2  # 第二个参数为gpu数量                          #
# GPU_IDS=$3  # 第三个参数为gpu_ids                      #
# -------------------------------------------------- #
GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))    # 最大8个gpu
NNODES=`expr $GPUS / $GPUS_PER_NODE`  # 计算节点数

# 训练的主节点的端口、主节点的地址和当前节点的排名
MASTER_PORT=${MASTER_PORT:-28596}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
RANK=${RANK:-0}
# GPU_IDS=(0 1 2)


# 生成work_dir, logs目录
WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/
# Intermediate files and logs will be saved to UniAD/projects/work_dirs/

#* 设置训练的检查点
# 第一阶段在 UniAD/projects/work_dirs/stage1_track_map/base_track_map/latest.pth
# 第二阶段在 UniAD/projects/work_dirs/stage2_e2e/base_e2e/latest.pth
RESUME_FROM="${WORK_DIR}latest.pth"  # 设置恢复训练的检查点路径M

echo "RESUME: ${RESUME_FROM}"
if [ ! -d ${WORK_DIR}logs ]; then
    mkdir -p ${WORK_DIR}logs
fi

# PYTHON 环境变量设置
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --nnodes=${NNODES} \
    --node_rank=${RANK} \
    $(dirname "$0")/train.py \
    $CFG \
    --launcher pytorch ${@:3} \
    --deterministic \
    --work-dir ${WORK_DIR} \
    --resume-from ${RESUME_FROM} \
    2>&1 | tee ${WORK_DIR}logs/train.$T