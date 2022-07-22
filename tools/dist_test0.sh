#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
#GPUS=$3
PORT=${PORT:-29500}

MKL_SERVICE_FORCE_INTEL=1 PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# Arguments starting from the forth one are captured by ${@:4}
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$PORT \
    $(dirname "$0")/test0.py $CONFIG -C $CHECKPOINT --launcher pytorch ${@:4}
