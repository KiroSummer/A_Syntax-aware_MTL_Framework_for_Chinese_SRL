export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda:/usr/local/cuda/lib64:/opt/OpenBLAS/lib

CONFIG="config.json"
MODEL="conll05_model"

TRAIN_PATH="../data/chn_srl/train.chinese.cpb1.0.jsonlines"
DEV_PATH="../data/chn_srl/dev.chinese.cpb1.0.jsonlines"
GOLD_PATH="../data/chn_srl/dev/dev.props"
DEP_TREES="../data/dependency-trees/ALL-from-xpeng/dep_all.conll"

CHARS="./char_vocab_cpb_all_dep.txt"

gpu_id=$1
CUDA_VISIBLE_DEVICES=$gpu_id python2 ../src/baseline-MTL-dep-private-lstm-weighted-sum-as-input/train.py \
   --config=$CONFIG \
   --model=$MODEL \
   --train=$TRAIN_PATH \
   --dev=$DEV_PATH \
   --gold=$GOLD_PATH \
   --dep_trees=$DEP_TREES \
   --chars=$CHARS \
   --gpu=$1
