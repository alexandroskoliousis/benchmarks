#!/bin/bash
#
# Benchmark(s) for TensorFlow.
#
PROFILE=0

NVPROFEXEC=/usr/local/cuda-10.1/bin/nvprof
NVPROFOPTS="-o profiler.nvvp --profile-from-start off --concurrent-kernels off"
if [ $PROFILE -gt 1 ]; then
	#
	# Detailed profiling looks into SM occupancy and activity.
	# Run only 1 batch on 1 GPU.
	#
	NVPROFOPTS=${NVPROFOPTS}" --metrics achieved_occupancy,sm_activity"
fi

# Batch size (per GPU)
B=32
# Numbe of GPU devices
G=1
# Learning rate schedule (step-wise)
S="0.1;80;0.01;120;0.001" # LR0;E1;LR1;...;En;LRn
# Version of ResNet
V="0"

# Read from the command line...
# B=$1
# G=$2
# S=$3


FLAGS=

#
# Basic configuration
#

MODEL="resnet32"

DATASET="cifar10"

# Synthetic data
# DATADIR=""
DATADIR="datasets/cifar-10/batches"
# TF record data
# DATADIR="datasets/cifar-10"

EPOCHS=5 # 160

MOMENTUM=0.9
DECAY="0.0001"
OPTIMISER="momentum" # "sgd" or "momentum"

# Synchronisation parameters

UPDATE="replicated" # parameter_server, replicated, independent
REPACKING="8"
STAGED="False"
ALLREDUCE="nccl" # empty, nccl, or xring

DISPLAY_INTERVAL=100

# Also checkpoint every N epochs
CHECKPOINT_EVERY_N_EPOCHS=False

# We want 1 checkpoint per epoch...
CHECKPOINT_INTERVAL=1
CHECKPOINT_DIRECTORY="checkpoints"

FLAGS=${FLAGS}" --model=${MODEL}"
FLAGS=${FLAGS}" --data_name=${DATASET}"

[ -n "${DATADIR}" ] && FLAGS=${FLAGS}" --data_dir=${DATADIR}"

if [ $PROFILE -gt 1 ]; then
	# When profiling SM activity,
	# run only 1 batch.
	FLAGS=${FLAGS}" --num_batches=1"
else
	FLAGS=${FLAGS}" --num_epochs=${EPOCHS}"
fi
# FLAGS=${FLAGS}" --num_batches=10"

# Run benchmark in training mode
FLAGS=${FLAGS}" --eval=False --forward_only=False"

# Print training accuracy
FLAGS=${FLAGS}" --print_training_accuracy=True" 

# Set default random seed
FLAGS=${FLAGS}" --tf_random_seed=123456789"

# CPU/GPU configuration
if [ $PROFILE -gt 1 ]; then
	# Overide configuration
	G=1
fi
G=1
FLAGS=${FLAGS}" --num_gpus=$G --gpu_thread_mode=gpu_private"

# No warm-up
FLAGS=${FLAGS}" --num_warmup_batches=0"

# Hyper-parameters
FLAGS=${FLAGS}" --batch_size=${B}"

# By default learning_rate is None, in which case we fall 
# back to a model-specific learning rate
# 
# By default `--piecewise_learning_rate_schedule` is None
[ -n "${S}" ] && FLAGS=${FLAGS}" --piecewise_learning_rate_schedule=${S}"

FLAGS=${FLAGS}" --momentum=${MOMENTUM}"
FLAGS=${FLAGS}" --weight_decay=${DECAY}"
FLAGS=${FLAGS}" --optimizer=${OPTIMISER}"

# Synchronisation parameters
FLAGS=${FLAGS}" --variable_update=${UPDATE}"
FLAGS=${FLAGS}" --staged_vars=${STAGED}"
[ -n "${ALLREDUCE}" ] && FLAGS=${FLAGS}" --all_reduce_spec=${ALLREDUCE}"

# Image producer configuration
FLAGS=${FLAGS}" --use_datasets=True"

# Image pre-processor configuration
FLAGS=${FLAGS}" --distortions=False"
FLAGS=${FLAGS}" --fuse_decode_and_crop=True"
FLAGS=${FLAGS}" --resize_method=bilinear"

# Checkpoints
#
# Fix display interval to every 100 tasks...
FLAGS=${FLAGS}" --display_every=${DISPLAY_INTERVAL}"

FLAGS=${FLAGS}" --checkpoint_every_n_epochs=False"
FLAGS=${FLAGS}" --checkpoint_interval=${CHECKPOINT_INTERVAL}"
FLAGS=${FLAGS}" --checkpoint_directory=${CHECKPOINT_DIRECTORY}"

FLAGS=${FLAGS}" --data_format=NCHW"
FLAGS=${FLAGS}" --batchnorm_persistent=True"
FLAGS=${FLAGS}" --use_tf_layers=True"
# FLAGS=${FLAGS}" --winograd_nonfused=True"
FLAGS=${FLAGS}" --winograd_nonfused=False"

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
    $MEASUREMENTSCRIPT "resnet-32-b-${B}-g-${G}.csv" &
    # Get background process id
    MEASUREMENTSCRIPTPID=$!
fi

# Run
printf "Train ResNet32 v.%s with batch size %4d on %d GPU devices\n" $V $B $G
#
if [ $PROFILE -gt 0 ]; then
	# Check if profiler is available
	which $NVPROFEXEC >/dev/null 2>&1
	if [ $? -eq 1 ]; then
		echo "error: $NVPROF not found"
		exit 1
	fi
	# Print useful info
	echo -n "Profiler mode: "
	if [ $PROFILE -gt 1 ]; then
		echo "Run 1 batch on 1 GPU"
	else
		echo "Train for $EPOCHS epochs"
	fi
	
	echo "$NVPROFEXEC $NVPROFOPTS python tf_cnn_benchmarks.py $FLAGS"
	$NVPROFEXEC $NVPROFOPTS python tf_cnn_benchmarks.py $FLAGS # > results/resnet-32-b-$B-g-$G.out
else
	python tf_cnn_benchmarks.py $FLAGS > resnet-32-b-${B}-g-${G}.out 2>&1
fi

if [ $MEASUREMENTS -gt 0 ]; then
    # Stop GPU measurements script
    echo "Stop GPU measurements"
    if [ -n $MEASUREMENTSCRIPTPID ]; then
        kill -15 $MEASUREMENTSCRIPTPID >/dev/null 2>&1
        killall "nvidia-smi" >/dev/null 2>&1
    fi
fi

echo "Done"
