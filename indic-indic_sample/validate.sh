#!/bin/sh

source ./common.sh

#model prefix
prefix=model/model.npz

dev=data/tun.subword.$SRC
ref=data/tun.$TRG
out=output/tun.subword.$TRG

# decode
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn python $nematus/nematus/translate.py \
     -m $prefix.dev.npz \
     -i $dev \
     -o $out \
     -k 12 -n -p 1

./postprocess-dev.sh < $out > $out.postprocessed


## get BLEU
BEST=`cat ${prefix}_best_bleu || echo 0`
$mosesdecoder/scripts/generic/multi-bleu.perl $ref < $out.postprocessed >> ${prefix}_bleu_scores
BLEU=`$mosesdecoder/scripts/generic/multi-bleu.perl $ref < $out.postprocessed | cut -f 3 -d ' ' | cut -f 1 -d ','`
BETTER=`echo "$BLEU > $BEST" | bc`

echo "BLEU = $BLEU"

# save model with highest BLEU
if [ "$BETTER" = "1" ]; then
  echo "new best; saving"
  echo $BLEU > ${prefix}_best_bleu
  cp ${prefix}.dev.npz ${prefix}.npz.best_bleu
fi
