#!/bin/sh

# theano device, in case you do not want to compute on gpu, change it to cpu
device=gpu1

SRC=en
TRG=hi

# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=/home/development/anoop/installs/nematus

## path to transliterator 
export XLIT_HOME=/home/development/anoop/experiments/multilingual_unsup_xlit/src/conll16_unsup_xlit
export PYTHONPATH=$PYTHONPATH:$XLIT_HOME/src

THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn python $nematus/nematus/translate.py \
     -m model/model.npz \
     -i data/test.$SRC \
     -o output/test.output.$TRG \
     -k 5 -n -p 4 --device-list gpu2 gpu3 \
     --n-best

out_moses_fname=output/test.output.mosesformat.$TRG
./postprocess-test.sh < output/test.output.$TRG > $out_moses_fname

# generate NEWS 2015 evaluation format output file 
python $XLIT_HOME/src/cfilt/transliteration/news2015_utilities.py gen_news_output \
        "data/test.id" \
        "data/test.xml" \
        "$out_moses_fname" \
        "$out_moses_fname.xml" \
        "system" "conll2016" "$SRC" "$TRG"  

# run evaluation 
python $XLIT_HOME/scripts/news_evaluation_script/news_evaluation.py \
        -t "data/test.xml" \
        -i "$out_moses_fname.xml" \
        -o "$out_moses_fname.detaileval.csv" \
         > "$out_moses_fname.eval"

