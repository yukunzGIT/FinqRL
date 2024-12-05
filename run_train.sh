TRAIN="data/train_set.txt"
BATCH=16
HDIM=200
NLAY=2
MIN_RR=0.5
MIN_CR=0.3
LR=0.001
SAVE="./model"

python ./src/trainer.py \
    -t $TRAIN \
    -batch $BATCH \
    -min_rr $MIN_RR \
    -min_cr $MIN_CR \
    -save $SAVE \
    -lr $LR \
    -hdim $HDIM \
    -nlayer $NLAY
