#!/bin/bash
#
# Modified on 16 April to run version 1.5 with NCCL support

VERSION="1.5"
DATADIR="/fast/imagenet/train"
CHECKPOINTDIR="/fast/checkpoints/"
# B=64
B=128
# B=256 causes out-of-memory error
# --xla_compile=True increases throughput but we don't support it.

python tf_cnn_benchmarks.py \
	--data_format=NCHW \
	--batch_size=${B} \
	--model=resnet50_v1.5 \
	--optimizer=momentum \
	--variable_update=replicated \
	--all_reduce_spec=nccl \
	--nodistortions \
	--gradient_repacking=2 \
	--num_gpus=8 \
	--num_epochs=91 \
	--num_warmup_batches=10 \
	--weight_decay=1e-4 \
	--data_dir=${DATADIR} \
	--checkpoint_interval=1 \
	--checkpoint_directory=${CHECKPOINTDIR} \
	--print_training_accuracy=True \
	--compute_lr_on_cpu=True \
	--single_l2_loss_op=True \
	--loss_type_to_report=base_loss \
	>resnet-50-r${VERSION}.out 2>&1

echo "Bye."
exit 0
