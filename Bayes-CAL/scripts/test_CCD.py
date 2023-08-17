#!/bin/bash

cd ..
CUDA_LAUNCH_BLOCKING=1

# custom config
DATA=/DATA  # /path/to/datasets
TRAINER=BCAL

DATASET=ColoredCatsDogs
CFG=rn50_ep30

python train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--eval-only \
--model-dir /output/ColoredCatsDogs/ColoredCatsDogs.json/vicoop/seed1/0.1311_3.3260_0.2966/rn50_ep20_16shots/nctx16_cscFalse_ctpmiddle \
--load-epoch 30
