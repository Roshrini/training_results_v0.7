#!/bin/bash
CONFIG='configs/e2e_mask_rcnn_R_50_FPN_1x.yaml'

#This folder should a file called 'last_checkpoint' which contains the path to the actual checkpoint
FOLDER='/shared/datasets/checkpoints_32_epoch17'
LOGFILE="$FOLDER/job_eval32_epoch.log"
if ! [ -d "$FOLDER" ]; then mkdir $FOLDER; fi

echo "Launching job"
#python3 -u -m bind_launch --nnodes 1 --node_rank 0 --master_addr 127.0.0.0 --master_port 1234 \
#                         --nsockets_per_node=2 \
# 			--ncores_per_socket=24 --nproc_per_node=8 \
python3 -m torch.distributed.launch --nproc_per_node=8 tools/test_net.py --config-file '/shared/roshanin/training_results_v0.7/NVIDIA/benchmarks/maskrcnn/implementations/pytorch/configs/e2e_mask_rcnn_R_50_FPN_1x.yaml' \
 			DTYPE 'float16' \
			PATHS_CATALOG '/shared/roshanin/training_results_v0.7/NVIDIA/benchmarks/maskrcnn/implementations/pytorch/maskrcnn_benchmark/config/paths_catalog_dbcluster.py' \
			OUTPUT_DIR $FOLDER \
 			DISABLE_REDUCED_LOGGING True \
			TEST.IMS_PER_BATCH 8 \
			NHWC True \
			| tee $LOGFILE

#2019-02-22 00:05:39,954 maskrcnn_benchmark.inference INFO: Total inference time: 0:04:55.840343 (0.05916806864738464 s / img per device, on 1 devices)

time=`cat $LOGFILE | grep -F 'maskrcnn_benchmark.inference INFO: Total inference time' | tail -n 1 | awk -F'(' '{print $2}' | awk -F' s ' '{print $1}' | egrep -o [0-9.]+`
calc=$(echo $time 1.0 | awk '{ printf "%f", $2 / $1 }')
