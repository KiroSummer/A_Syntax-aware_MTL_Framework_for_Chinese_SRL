#!/bin/bash

SRL_PATH="./data/chn_srl"

if [ ! -d $SRL_PATH ]; then
  mkdir -p $SRL_PATH
fi

EMB_PATH="./data/embeddings"
if [ ! -d $EMB_PATH ]; then
  mkdir -p $EMB_PATH
fi

# process the cpb1.0 data
CPB_PATH="./data/chn-srl.1.0.tgz"

tar -zxvf $CPB_PATH -C ${SRL_PATH}/

cd ${SRL_PATH}

# Prepare train/dev/test set.
paste -d ' ' ./trn/trn.sem-synt ./trn/trn.props > "train-set"
paste -d ' ' ./dev/dev.sem-synt ./dev/dev.props > "dev-set"
paste -d ' ' ./tst/tst.sem-synt ./tst/tst.props  > "test-set"

cd ../..

# Convert CoNLL to json format.
python scripts/cpb_to_json.py data/chn_srl/train-set ./data/chn_srl/train.chinese.cpb1.0.jsonlines 4
python scripts/cpb_to_json.py data/chn_srl/dev-set ./data/chn_srl/dev.chinese.cpb1.0.jsonlines 4
python scripts/cpb_to_json.py data/chn_srl/test-set ./data/chn_srl/test.chinese.cpb1.0.jsonlines 4

# Filter embeddings.
python scripts/filter_embeddings.py ${EMB_PATH}/giga.demo.0.7.word.emb.txt \
  ${EMB_PATH}/giga.demo.0.7.word.emb.txt.CPB1.0.filtered \
  ${SRL_PATH}/train.chinese.cpb1.0.jsonlines ${SRL_PATH}/dev.chinese.cpb1.0.jsonlines


