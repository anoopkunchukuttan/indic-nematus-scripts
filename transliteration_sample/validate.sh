#!/bin/sh

# suffix of source language files
SRC=en

# suffix of target language files
TRG=hi

# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=/home/development/anoop/installs/nematus

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=/usr/local/bin/smt/mosesdecoder_29July16/

# theano device, in case you do not want to compute on gpu, change it to cpu
device=gpu

#model prefix
prefix=model/model.npz

dev=data/tun.$SRC
ref=data/tun.$TRG

# decode
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn python $nematus/nematus/translate.py \
     -m $prefix.dev.npz \
     -i $dev \
     -o output/tun.output.$TRG \
     --n-best \
     -k 5 -n -p 1

### get BLEU
#BEST=`cat ${prefix}_best_bleu || echo 0`
#$mosesdecoder/scripts/generic/multi-bleu.perl $ref < $dev.output.postprocessed.dev >> ${prefix}_bleu_scores
#BLEU=`$mosesdecoder/scripts/generic/multi-bleu.perl $ref < $dev.output.postprocessed.dev | cut -f 3 -d ' ' | cut -f 1 -d ','`
#BETTER=`echo "$BLEU > $BEST" | bc`
#
#echo "BLEU = $BLEU"
#
## save model with highest BLEU
#if [ "$BETTER" = "1" ]; then
#  echo "new best; saving"
#  echo $BLEU > ${prefix}_best_bleu
#  cp ${prefix}.dev.npz ${prefix}.npz.best_bleu
#fi
