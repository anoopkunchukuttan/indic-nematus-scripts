
# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=/usr/local/bin/smt/mosesdecoder_29July16/

# path to subword segmentation scripts: https://github.com/rsennrich/subword-nmt
subword_nmt=/home/development/anoop/installs/subword-nmt

# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=/home/development/anoop/installs/nematus
export PYTHONPATH=$PYTHONPATH:/home/development/anoop/installs/nematus

# suffix of source language files
SRC=bn

# suffix of target language files
TRG=hi

# theano device, in case you do not want to compute on gpu, change it to cpu
device=cuda

# network vocab size 
VOCAB_SIZE = 3500
# number of merge operations. Network vocabulary should be slightly larger (to include characters),

# number of merge operations. Network vocabulary should be slightly larger (to include characters),
# or smaller if the operations are learned on the joint vocabulary
bpe_operations=3000

## export variables 
export SRC
export TRG
export VOCAB_SIZE
