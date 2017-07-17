#!/bin/sh

# suffix of source language files
SRC=en

# suffix of target language files
TRG=hi

# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=/home/development/anoop/installs/nematus

## path to transliterator 
export XLIT_HOME=/home/development/anoop/experiments/multilingual_unsup_xlit/src/conll16_unsup_xlit
export PYTHONPATH=$PYTHONPATH:$XLIT_HOME/src

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
     -k 5 -n -p 1 \
     --n-best

out_moses_fname=output/tun.output.mosesformat.$TRG
./postprocess-dev.sh < output/tun.output.$TRG > $out_moses_fname

# generate NEWS 2015 evaluation format output file 
python $XLIT_HOME/src/cfilt/transliteration/news2015_utilities.py gen_news_output \
        "data/tun.id" \
        "data/tun.xml" \
        "$out_moses_fname" \
        "$out_moses_fname.xml" \
        "system" "conll2016" "$SRC" "$TRG"  

# run evaluation 
python $XLIT_HOME/scripts/news_evaluation_script/news_evaluation.py \
        -t "data/tun.xml" \
        -i "$out_moses_fname.xml" \
        -o "$out_moses_fname.detaileval.csv" \
         > "$out_moses_fname.eval"
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
