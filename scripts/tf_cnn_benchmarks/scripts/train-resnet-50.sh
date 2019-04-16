#!/bin/bash
#
# Modified on 16 April to run version 1.5 with NCCL support

VERSION="1.5"
DATADIR="/fast/imagenet/train"
CHECKPOINTDIR="/fast/checkpoints/"

python tf_cnn_benchmarks.py \
	--data_format=NCHW \
	--batch_size=64 \
	--model=resnet50_v1.5 \
	--optimizer=momentum \
	--variable_update=replicated \
	--all_reduce_spec=nccl \
	--nodistortions \
	--gradient_repacking=2 \
	--num_gpus=8 \
	--num_epochs=91 \
	--num_warmup_batches=0 \
	--weight_decay=1e-4 \
	--data_dir=${DATADIR} \
	--checkpoint_interval=1 \
	--checkpoint_directory=${CHECKPOINTDIR} \
	--print_training_accuracy=True \
	>resnet-50-r${VERSION}.out 2>&1

echo "Bye."
exit 0
