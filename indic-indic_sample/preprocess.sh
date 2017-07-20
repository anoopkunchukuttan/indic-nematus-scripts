#!/bin/sh

# this sample script preprocesses a sample corpus, including tokenization,
# truecasing, and subword segmentation. 
# for application to a different language pair,
# change source and target prefix, optionally the number of BPE operations,
# and the file names (currently, data/corpus and data/newsdev2016 are being processed)

# in the tokenization step, you will want to remove Romanian-specific normalization / diacritic removal,
# and you may want to add your own.
# also, you may want to learn BPE segmentations separately for each language,
# especially if they differ in their alphabet

# suffix of source language files
SRC=bn

# suffix of target language files
TRG=hi

# number of merge operations. Network vocabulary should be slightly larger (to include characters),
# or smaller if the operations are learned on the joint vocabulary
bpe_operations=3000

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=/usr/local/bin/smt/mosesdecoder_29July16/

# path to subword segmentation scripts: https://github.com/rsennrich/subword-nmt
subword_nmt=/home/development/anoop/installs/subword-nmt

# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=/home/development/anoop/installs/nematus

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
parallel --gnu --colsep '-' "$subword_nmt/apply_bpe.py -c model/{2}.bpe < data/{1}.{2} > data/{1}.bpe.{2}"

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
#parallel --gnu --colsep '-' "$subword_nmt/apply_bpe.py -c model/$SRC$TRG.bpe < data/{1}.{2} > data/{1}.bpe.{2}"
#
###### joint BPE operations END #######

## build network dictionary
$nematus/data/build_dictionary.py data/train.bpe.$SRC data/train.bpe.$TRG
