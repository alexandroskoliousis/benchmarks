#!/bin/bash

VERSION="1"
DATADIR="/fast/tensorflow/imagenet/train"
CHECKPOINTDIR="/fast/tensorflow/checkpoints/"

# B=1
# B=2
# B=4
B=8
# B=16
# B=32
# B=64

G=8

# Enable GPU utilisation measurements
MEASUREMENTS=1
MEASUREMENTSCRIPT="$CROSSBOW_HOME/tools/measurements/gpu-measurements.sh"
MEASUREMENTSCRIPTPID=

if [ $MEASUREMENTS -gt 0 ]; then
    if [ ! -x $MEASUREMENTSCRIPT ]; then
        echo "error: invalid script: $MEASUREMENTSCRIPT"
        exit 1
    fi
    # Let's generate an appropriate filename
    # to store measurements
    $MEASUREMENTSCRIPT "resnet-101-b-${B}-g-${G}-m-1.csv" &
    # Get background process id
    MEASUREMENTSCRIPTPID=$!
    # echo "Measurement process ID is $MEASUREMENTSCRIPTPID"
fi

if [ $MEASUREMENTS -gt 0 ]; then
    # When measuring GPU utilisation, run
    # only for a few 100s of tasks and do
    # checkpoint the model.
    python tf_cnn_benchmarks.py \
        --data_format=NCHW \
        --batch_size=64 \
        --model=resnet101 \
        --optimizer=momentum \
        --variable_update=replicated \
        --all_reduce_spec=nccl \
        --nodistortions \
        --gradient_repacking=8 \
        --num_gpus=${G} \
        --num_batches=1000 \
        --weight_decay=1e-4 \
        --data_dir=${DATADIR} \
        --checkpoint_interval=1 \
        --checkpoint_directory=${CHECKPOINTDIR} \
        --checkpoint_every_n_epochs=False \
        --print_training_accuracy=True \
        >resnet-101-r${VERSION}-b-${B}-g-${G}.out 2>&1
else
    # GPU measurements are disabled: train for convergence
    python tf_cnn_benchmarks.py \
        --data_format=NCHW \
        --batch_size=${B} \
        --model=resnet101 \
        --optimizer=momentum \
        --variable_update=replicated \
        --all_reduce_spec=nccl \
        --nodistortions \
        --gradient_repacking=8 \
        --num_gpus=8 \
        --num_epochs=90 \
        --weight_decay=1e-4 \
        --data_dir=${DATADIR} \
        --checkpoint_interval=1 \
        --checkpoint_directory=${CHECKPOINTDIR} \
        --print_training_accuracy=True \
        >resnet-101-r${VERSION}.out 2>&1
    
fi

if [ $MEASUREMENTS -gt 0 ]; then
    # Stop GPU measurements script
    echo "Stop GPU measurements"
    if [ -n $MEASUREMENTSCRIPTPID ]; then
        # Kill process
        kill -15 $MEASUREMENTSCRIPTPID >/dev/null 2>&1
        # Temporary solution until we can kill
        # child process safely
        killall "nvidia-smi" >/dev/null 2>&1
    fi
fi

echo "Bye."
exit 0
