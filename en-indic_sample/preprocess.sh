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
SRC=en

# suffix of target language files
TRG=hi

# number of merge operations. Network vocabulary should be slightly larger (to include characters),
# or smaller if the operations are learned on the joint vocabulary
bpe_operations=89500

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=/usr/local/bin/smt/mosesdecoder_29July16/

# path to subword segmentation scripts: https://github.com/rsennrich/subword-nmt
subword_nmt=/home/development/anoop/installs/subword-nmt

# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=/home/development/anoop/installs/nematus

# tokenize
for prefix in train test tun
do
   cp data/$prefix.$SRC data/$prefix.tok.$SRC
   cp data/$prefix.$TRG data/$prefix.tok.$TRG
done

# clean empty and long sentences, and sentences with high source-target ratio (training corpus only)
$mosesdecoder/scripts/training/clean-corpus-n.perl data/train.tok $SRC $TRG data/train.tok.clean 1 80

# train truecaser
$mosesdecoder/scripts/recaser/train-truecaser.perl -corpus data/train.tok.clean.$SRC -model model/truecase-model.$SRC

# apply truecaser (cleaned training corpus)
for prefix in train
 do
  $mosesdecoder/scripts/recaser/truecase.perl -model model/truecase-model.$SRC < data/$prefix.tok.clean.$SRC > data/$prefix.tc.$SRC
  cp data/$prefix.tok.clean.$TRG data/$prefix.tc.$TRG 
 done

# apply truecaser (dev/test files)
for prefix in tun test
 do
  $mosesdecoder/scripts/recaser/truecase.perl -model model/truecase-model.$SRC < data/$prefix.tok.$SRC > data/$prefix.tc.$SRC
  cp data/$prefix.tok.$TRG data/$prefix.tc.$TRG
 done

# train BPE
#cat data/train.tc.$SRC data/train.tc.$TRG | $subword_nmt/learn_bpe.py -s $bpe_operations > model/$SRC$TRG.bpe
cat data/train.tc.$SRC | $subword_nmt/learn_bpe.py -s $bpe_operations > model/$SRC.bpe
cat data/train.tc.$TRG | $subword_nmt/learn_bpe.py -s $bpe_operations > model/$TRG.bpe

# apply BPE

for prefix in train tun test
 do
  $subword_nmt/apply_bpe.py -c model/$SRC.bpe < data/$prefix.tc.$SRC > data/$prefix.bpe.$SRC
  $subword_nmt/apply_bpe.py -c model/$TRG.bpe < data/$prefix.tc.$TRG > data/$prefix.bpe.$TRG
 done

# build network dictionary
$nematus/data/build_dictionary.py data/train.bpe.$SRC data/train.bpe.$TRG
