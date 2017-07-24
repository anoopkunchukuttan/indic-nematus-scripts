import numpy
import os
import sys
import logging

VOCAB_SIZE = int(os.getenv('VOCAB_SIZE')
SRC = os.getenv('SRC')
TRG = os.getenv('TRG')
DATA_DIR = "data/"

from nematus.nmt import train

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

if __name__ == '__main__':
    validerr = train(saveto='model/model.npz',
                    reload_=True,
                    dim_word=256,
                    dim=512,
                    n_words=VOCAB_SIZE,
                    n_words_src=VOCAB_SIZE,
                    decay_c=0.,
                    clip_c=1.,
                    lrate=0.0001,
                    optimizer='adam',
                    maxlen=50,
                    batch_size=50,
                    valid_batch_size=50,
                    datasets=[DATA_DIR + '/train.subword.' + SRC, DATA_DIR + '/train.subword.' + TRG],
                    valid_datasets=[DATA_DIR + '/tun.subword.' + SRC, DATA_DIR + '/tun.subword.' + TRG],
                    dictionaries=[DATA_DIR + '/train.subword.' + SRC + '.json',DATA_DIR + '/train.subword.' + TRG + '.json'],
                    validFreq=5000,
                    dispFreq=1000,
                    saveFreq=5000,
                    sampleFreq=10000,
                    use_dropout=False,
                    dropout_embedding=0.2, # dropout for input embeddings (0: no dropout)
                    dropout_hidden=0.2, # dropout for hidden layers (0: no dropout)
                    dropout_source=0.1, # dropout source words (0: no dropout)
                    dropout_target=0.1, # dropout target words (0: no dropout)
                    patience=5,
                    overwrite=False,
                    external_validation_script='./validate.sh')
    print validerr
