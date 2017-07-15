#!/bin/sh

# theano device, in case you do not want to compute on gpu, change it to cpu
device=gpu

SRC=en
TRG=hi

# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=/home/development/anoop/installs/nematus

THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn python $nematus/nematus/translate.py \
     -m model/model.npz \
     -i data/test.$SRC \
     -o data/test.output.$TRG \
     -k 12 -n -p 1


./postprocess-test.sh < data/test.output.$TRG > data/test.output.postprocessed.$TRG


## get BLEU
BEST=`cat ${prefix}_best_bleu || echo 0`
$mosesdecoder/scripts/generic/multi-bleu.perl $ref < data/test.output.postprocessed.$TRG > bleu_score.txt
