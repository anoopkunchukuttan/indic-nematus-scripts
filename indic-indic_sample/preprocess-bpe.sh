#!/bin/sh

source ./common.sh

# clean empty and long sentences, and sentences with high source-target ratio (training corpus only)
$mosesdecoder/scripts/training/clean-corpus-n.perl data/train $SRC $TRG data/train.clean 1 80

##### separate BPE operations #######

# train BPE
cat data/train.clean.$SRC | $subword_nmt/learn_bpe.py -s $bpe_operations > model/$SRC.bpe
cat data/train.clean.$TRG | $subword_nmt/learn_bpe.py -s $bpe_operations > model/$TRG.bpe

# apply BPE
for prefix in train.clean tun test
do
    for lang in $SRC $TRG
    do 
        echo $prefix-$lang
    done 
done | \
parallel --gnu --colsep '-' "$subword_nmt/apply_bpe.py -c model/{2}.bpe < data/{1}.{2} > data/{1}.subword.{2}"

##### separate BPE operations END #######

###### joint BPE operations #######
#
## train BPE
#cat data/train.clean.$SRC data/train.clean.$TRG | $subword_nmt/learn_bpe.py -s $bpe_operations > model/$SRC$TRG.bpe
#
## apply BPE
#for prefix in train.clean tun test
#do
#    for lang in $SRC $TRG
#    do 
#        echo $prefix-$lang
#    done 
#done | \
#parallel --gnu --colsep '-' "$subword_nmt/apply_bpe.py -c model/$SRC$TRG.bpe < data/{1}.{2} > data/{1}.subword.{2}"
#
###### joint BPE operations END #######

## build network dictionary
$nematus/data/build_dictionary.py data/train.subword.$SRC data/train.subword.$TRG
