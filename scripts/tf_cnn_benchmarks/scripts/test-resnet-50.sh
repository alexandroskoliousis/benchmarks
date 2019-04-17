#!/bin/bash
#
# Benchmark(s) for TensorFlow.
#
# VERSION=`python helpers/findcheckpoint.py results/b-${b}-g-${g}.out`
# printf "Test results from batch size %4d on %d GPU devices (v-%s)\n" $b $g $VERSION
VERSION="000002"

FLAGS=

#
# Basic configuration
#
# Run benchmark in training mode
EVALUATION="True"

MODEL="resnet50_v1.5"

NAME="imagenet"
# Synthetic data
DATADIR=""
# TF record data
# DATADIR="/mnt/LSDSDataShare/projects/16-crossbow/platypus2/data/tf/imagenet/records"
DATADIR="/fast/imagenet/validation"

# Only one of the two must be set
EPOCHS=1

FLAGS="${FLAGS} --eval=${EVALUATION} --forward_only=False"

FLAGS="${FLAGS} --model=${MODEL}"
FLAGS="${FLAGS} --data_name=${NAME}"

[ -n "${DATADIR}" ] && FLAGS="${FLAGS} --data_dir=${DATADIR}"

FLAGS="${FLAGS} --num_epochs=${EPOCHS}"

# Defaults

# Set default random seed
FLAGS="${FLAGS} --tf_random_seed=1234"

#
# CPU/GPU configuration
#
NGPU=8

FLAGS="${FLAGS} --num_gpus=${NGPU}"

#
# Hyper-parameters
#
BATCH_SIZE=50

WEIGHT_DECAY="1e-4"
# Configure optimiser
# HAS_MOMENTUM=`echo "$MOMENTUM > 0.0" | bc -l`
HAS_MOMENTUM=1
if [ $HAS_MOMENTUM -eq 0 ]; then
    OPTIMISER="sgd"
else
    OPTIMISER="momentum"
fi

FLAGS=${FLAGS}" --batch_size=${BATCH_SIZE}"

FLAGS=${FLAGS}" --weight_decay=${WEIGHT_DECAY}"
FLAGS=${FLAGS}" --optimizer=${OPTIMISER}"

# Defaults

#
# Synchronisation parameters
#
VARIABLE_UPDATE="replicated" # parameter_server, replicated, independent

FLAGS="${FLAGS} --variable_update=${VARIABLE_UPDATE}"

#
# Checkpoints, summaries, etc.
#
DISPLAY_INTERVAL=10

CHECKPOINT_EVERY_N_EPOCHS="True"
CHECKPOINT_INTERVAL=1
CHECKPOINT_DIRECTORY="/fast/checkpoints/v-${VERSION}"

FLAGS="${FLAGS} --checkpoint_every_n_epochs=${CHECKPOINT_EVERY_N_EPOCHS}"
FLAGS="${FLAGS} --checkpoint_interval=${CHECKPOINT_INTERVAL}"
FLAGS="${FLAGS} --checkpoint_directory=${CHECKPOINT_DIRECTORY}"

FLAGS="${FLAGS} --data_format=NCHW"

# AttributeError: '_ThreadPoolDataset' object has no attribute 'make_initializable_iterator'
FLAGS="${FLAGS} --use_datasets=False"

# Run.
#
echo "python tf_cnn_benchmarks.py $FLAGS"
python tf_cnn_benchmarks.py $FLAGS

