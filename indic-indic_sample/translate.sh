#!/bin/sh

source ./common.sh

THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn python $nematus/nematus/translate.py \
     -m model/model.npz \
     -i data/test.bpe.$SRC \
     -o output/test.bpe.output.$TRG \
     -k 12 -n -p  1


./postprocess-test.sh < output/test.bpe.output.$TRG > output/test.bpe.output.postprocessed.$TRG


## get BLEU
$mosesdecoder/scripts/generic/multi-bleu.perl data/test.$TRG < output/test.bpe.output.postprocessed.$TRG > output/bleu_score.txt
