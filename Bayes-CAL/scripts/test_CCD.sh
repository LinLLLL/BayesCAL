#!/bin/bash
# /data_disk/zl/BCAL_OUTPUT/OUTPUT_v1/ColoredCatsDogs/ColoredCatsDogs.json/vicoop/seed2/0.9037_7.8642_0.1139/rn50_ep20_16shots/nctx16_cscFalse_ctpend
# /data_disk/zl/BCAL_OUTPUT/OUTPUT_v1/ColoredCatsDogs/ColoredCatsDogs.json/vicoop/seed1/0.1311_3.3260_0.2966/rn50_ep20_16shots/nctx16_cscFalse_ctpmiddle
cd ..
CUDA_LAUNCH_BLOCKING=1
export TF_ENABLE_ONEDNN_OPTS=0

# custom config
DATA=/home/zl/DATA  # /path/to/datasets
TRAINER=BCAL

DATASET=ColoredCatsDogs
CFG=rn50_ep20
TEST_ENV=ColoredCatsDogs.json

python train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--eval-only \
--model-dir /data_disk/zl/BCAL_OUTPUT/OUTPUT_v1/ColoredCatsDogs/ColoredCatsDogs.json/vicoop/seed2/0.9037_7.8642_0.1139/rn50_ep20_16shots/nctx16_cscFalse_ctpend \
--load-epoch 30 \
DATASET.NUM_SHOTS 16 \
TEST_ENV ${TEST_ENV} 
