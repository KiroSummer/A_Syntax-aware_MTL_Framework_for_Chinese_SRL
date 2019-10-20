# A Syntax-aware Multi-task Learning Framework for Chinese Semantic Role Labeling
This repository contains codes and configs for training models presented in : [A Syntax-aware Multi-task Learning Framework for Chinese Semantic Role Labeling](XXX)

## Data
If you have the CPB1.0 and CoNLL-2009 Chinese data, you can convert the original format into the json format using the code in xxx.

To use our code, you should install pytorch >= 0.4.0.

## Train
To train our neural models, you should set the train.sh and config.json. Then, run
```bash
nohup ./train.sh 0 > log.txt 2>&1 &
```
where 0 is the GPU id.
We also give an corresponding example in exp-baseline-MTL-IIR/

## Test
For test, you should run the predict.sh presentated in the exp-baseline-MTL-IIR/ dir.
```bash
./predict.sh 0
```

