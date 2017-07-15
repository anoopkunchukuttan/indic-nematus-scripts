# theano device, in case you do not want to compute on gpu, change it to cpu
device=gpu

export PYTHONPATH=$PYTHONPATH:/home/development/anoop/installs/nematus

THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn python config.py

#nvcc.flags=-D_FORCE_INLINES
