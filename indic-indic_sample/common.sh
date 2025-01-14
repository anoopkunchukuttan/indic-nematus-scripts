
# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=/usr/local/bin/smt/mosesdecoder_18mar2017

# path to subword segmentation scripts: https://github.com/rsennrich/subword-nmt
subword_nmt=/home/development/anoop/installs/subword-nmt

# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=/home/development/anoop/installs/nematus
export PYTHONPATH=$PYTHONPATH:/home/development/anoop/installs/nematus

# short 
SUBWORD_SCRIPTS=/home/development/anoop/experiments/smt_phonetic/src/scripts

# path to parallel corpus 
ilci_corpus_path=/home/development/anoop/experiments/smt_phonetic/representation_unit/data/parallel/word

# suffix of source language files
SRC=bn

# suffix of target language files
TRG=hi

# theano device, in case you do not want to compute on gpu, change it to cpu
device=cuda

# network vocab size 
VOCAB_SIZE=3500

# number of merge operations. Network vocabulary should be slightly larger (to include characters),
# or smaller if the operations are learned on the joint vocabulary
bpe_operations=3000

ilci_os_corpus_path=/home/development/anoop/experiments/smt_phonetic/representation_unit/data/parallel/orth_split

## export variables 
export SRC
export TRG
export VOCAB_SIZE
