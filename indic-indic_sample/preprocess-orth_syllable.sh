#!/bin/sh

source ./common.sh

## copy orth split from directory and convert separator to nematus format 
cp $ilci_os_corpus_path/$SRC-$TRG/train.$SRC data/train.$SRC
cp $ilci_os_corpus_path/$SRC-$TRG/train.$TRG data/train.$TRG

cp $ilci_os_corpus_path/$SRC-$TRG/test.$SRC data/test.$SRC
cp $ilci_os_corpus_path/$SRC-$TRG/test.$TRG.split data/test.$TRG

cp $ilci_os_corpus_path/$SRC-$TRG/tun.$SRC data/tun.$SRC
cp $ilci_os_corpus_path/$SRC-$TRG/tun.$TRG.split data/tun.$TRG

# convert separator to internal marker format
for prefix in train tun test
do
    for lang in $SRC $TRG
    do 
        echo $prefix-$lang
    done 
done | \
parallel --dry-run --gnu --colsep '-' "$SUBWORD_SCRIPTS/utilities.py format_converter data/{1}.{2} data/{1}.tmp.{2} space_to_internal_marker_format '@' ; \
                            cat data/{1}.tmp.{2} | sed 's,@,@@,g'  >  data/{1}.subword.{2}"

## clean empty and long sentences, and sentences with high source-target ratio (training corpus only)
cp data/train.subword.$SRC data/train.all.$SRC 
cp data/train.subword.$TRG data/train.all.$TRG 

$mosesdecoder/scripts/training/clean-corpus-n.perl data/train.all $SRC $TRG data/train.subword 1 80

## build network dictionary
$nematus/data/build_dictionary.py data/train.subword.$SRC data/train.subword.$TRG
