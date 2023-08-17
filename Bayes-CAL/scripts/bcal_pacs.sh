#!/bin/bash

cd ..
#CUDA_VISIBLE_DEVICES=2
CUDA_LAUNCH_BLOCKING=1

# custom config
DATA=/DATA  # /path/to/datasets
TRAINER=BCAL

DATASET=$1
CFG=$2  # config file
CTP=$3  # class token position (end or middle)
NCTX=16
CSC=$4  # class-specific context (False or True)
alpha1=$5
alpha2=$6
alpha3=$7
SEED=$8
SHOTS=16

for TEST_ENV in "test_on_photo.json"  "test_on_cartoon.json"  "test_on_art_painting.json"  "test_on_sketch.json"
do
DIR=/output/${DATASET}/${TEST_ENV}/${TRAINER}/seed${SEED}/${alpha1}_${alpha2}_${alpha3}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/
if [ -d "$DIR" ]; then
echo "Results are available in ${DIR}. Skip this job"
else
echo "Run this job and save the output to ${DIR}"
python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${DIR} \
TRAINER.COOP.N_CTX ${NCTX} \
TRAINER.COOP.CSC ${CSC} \
TRAINER.COOP.CTX_INIT "a photo of a" \
TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
TRAINER.COCOOP.N_CTX ${NCTX} \
TRAINER.COCOOP.CSC ${CSC} \
TRAINER.COCOOP.CLASS_TOKEN_POSITION ${CTP} \
TRAINER.COCOOP.CTX_INIT "a photo of a" \
TRAINER.BCAL.N_CTX ${NCTX} \
TRAINER.BCAL.CSC ${CSC} \
TRAINER.BCAL.CLASS_TOKEN_POSITION ${CTP} \
TRAINER.BCAL.CTX_INIT "a photo of a" \
TRAINER.CONVENTION.N_CTX ${NCTX} \
TRAINER.CONVENTION.CSC ${CSC} \
TRAINER.CONVENTION.CLASS_TOKEN_POSITION ${CTP} \
TRAINER.CONVENTION.CTX_INIT "a photo of a" \
DATASET.NUM_SHOTS ${SHOTS} \
TEST_ENV ${TEST_ENV} \
alpha1 ${alpha1} \
alpha2 ${alpha2} \
alpha3 ${alpha3}
fi
done