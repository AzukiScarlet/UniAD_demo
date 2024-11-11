#!/usr/bin/env bash

T=`date +%m%d%H%M`

# -------------------------------------------------- #
# Usually you only need to customize these variables #
CFG=$1                                               #
CKPT=$2                                              #
GPUS=$3                                              #    
# GPU_IDS=$4                                           #
# -------------------------------------------------- #
GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))

MASTER_PORT=${MASTER_PORT:-28596}
WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/
# Intermediate files and logs will be saved to UniAD/projects/work_dirs/

# 用日期来区分.pkl文件
# OUTPUT_PATH=output/$(basename $CFG .py)/$(basename $CKPT .pth)/$T.pkl
OUTPUT_PATH=experiments/origin/stage2/origin/test/$T.pkl
SHOW_DIR=experiments/origin/stage2/origin/test/
echo "OUTPUT: ${OUTPUT_PATH}"
echo "SHOW_DIR: ${SHOW_DIR}"

if [ ! -d ${WORK_DIR}logs ]; then
    mkdir -p ${WORK_DIR}logs
fi
if [ ! -d ${SHOW_DIR}logs ]; then
    mkdir -p ${SHOW_DIR}logs
fi

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_port=$MASTER_PORT \
    $(dirname "$0")/test.py \
    $CFG \
    $CKPT \
    --launcher pytorch ${@:4} \
    --eval bbox \
    --out $OUTPUT_PATH \
    --show-dir ${SHOW_DIR} \
    2>&1 | tee ${SHOW_DIR}logs/eval.$T