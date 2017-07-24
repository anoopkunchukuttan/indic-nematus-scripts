#!/bin/bash

source ./common.sh

THRESHOLD=3

run_status=1
restart_count=0

while [  $run_status -eq 1   -a  $restart_count -lt  $THRESHOLD ]
do 
    echo 

    THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn \
        python config.py

    run_status=$?
    echo Run Status: $run_status

    if [ $run_status -eq 1 ]
    then 
        echo 'Command Failed'
        restart_count=$((restart_count+1))
        echo Ran $restart_count times unsuccessfully
        echo Waiting 
        sleep 10
        echo Restarting
    fi 

done 

if [ $run_status -eq 1 -a $restart_count -eq $THRESHOLD ]
then 
    echo "Ran unsuccessfuly $restart_count consecutive times.. Please check"
fi 

#nvcc.flags=-D_FORCE_INLINES
