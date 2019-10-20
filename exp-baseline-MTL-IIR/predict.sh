#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda:/usr/local/cuda/lib64:/opt/OpenBLAS/lib

MODEL_PATH="./conll05_model"

INPUT_PATH="../data/chn_srl/dev.chinese.cpb1.0.jsonlines"
GOLD_PATH="../data/chn_srl/dev/dev.props"
OUTPUT_PATH="../temp/cpb.devel.out"

#INPUT_PATH="../data/chn_srl/test.chinese.cpb1.0.jsonlines"
#GOLD_PATH="../data/chn_srl/tst/tst.props"
#OUTPUT_PATH="../temp/cpb.test.out"

CUDA_VISIBLE_DEVICES=$1 python2 ../src/baseline-MTL-dep-private-lstm-weighted-sum-as-input/predict.py \
  --model="$MODEL_PATH" \
  --input="$INPUT_PATH" \
  --output="$OUTPUT_PATH" \
  --gold="$GOLD_PATH" \
  --gpu=$1

