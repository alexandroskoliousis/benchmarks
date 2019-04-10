#!/bin/bash
#
# Benchmark(s) for TensorFlow.
#

# Batch size (per GPU)
B=32
# Numbe of GPU devices
G=4
# Learning rate schedule (step-wise)
S="0.1;80;0.01;120;0.001" # LR0;E1;LR1;...;En;LRn
# Version of ResNet
V="0"

# Read from the command line...
# B=$1
# G=$2
# S=$3

printf "Train ResNet32 v.%s with batch size %4d on %d GPU devices\n" $V $B $G

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

EPOCHS=160

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
# CHECKPOINT_EVERY_N_EPOCHS=True

# We want 1 checkpoint per epoch...
CHECKPOINT_INTERVAL=1
CHECKPOINT_DIRECTORY="checkpoints"

FLAGS=${FLAGS}" --model=${MODEL}"
FLAGS=${FLAGS}" --data_name=${DATASET}"

[ -n "${DATADIR}" ] && FLAGS=${FLAGS}" --data_dir=${DATADIR}"

FLAGS=${FLAGS}" --num_epochs=${EPOCHS}"

# Run benchmark in training mode
FLAGS=${FLAGS}" --eval=False --forward_only=False"

# Print training accuracy
FLAGS=${FLAGS}" --print_training_accuracy=True" 

# Set default random seed
FLAGS=${FLAGS}" --tf_random_seed=123456789"

# CPU/GPU configuration
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

FLAGS=${FLAGS}" --checkpoint_every_n_epochs=True"
FLAGS=${FLAGS}" --checkpoint_interval=${CHECKPOINT_INTERVAL}"
FLAGS=${FLAGS}" --checkpoint_directory=${CHECKPOINT_DIRECTORY}"

FLAGS=${FLAGS}" --data_format=NCHW"
FLAGS=${FLAGS}" --batchnorm_persistent=True"
FLAGS=${FLAGS}" --use_tf_layers=True"
FLAGS=${FLAGS}" --winograd_nonfused=True"

# Run
#
python tf_cnn_benchmarks.py $FLAGS # > results/resnet-32-b-$B-g-$G.out
