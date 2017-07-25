#!/bin/sh

source ./common.sh

THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn python $nematus/nematus/translate.py \
     -m model/model.npz \
     -i data/test.subword.$SRC \
     -o output/test.subword.output.$TRG \
     -k 12 -n -p  1


./postprocess-test.sh < output/test.subword.output.$TRG > output/test.subword.output.postprocessed.$TRG


## get BLEU
$mosesdecoder/scripts/generic/multi-bleu.perl $ilci_corpus_path/$SRC-$TRG/test.$TRG < output/test.subword.output.postprocessed.$TRG > output/bleu_score.txt
