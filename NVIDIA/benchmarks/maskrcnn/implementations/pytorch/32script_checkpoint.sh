#!/bin/bash

BASE_LR=0.24
MAX_ITER=40000
WARMUP_FACTOR=0.000133
WARMUP_ITERS=1800
STEPS="(5625,7500)"
TRAIN_IMS_PER_BATCH=256
TEST_IMS_PER_BATCH=256
FPN_POST_NMS_TOP_N_TRAIN=1000
NSOCKETS_PER_NODE=2
NCORES_PER_SOCKET=24
NPROC_PER_NODE=8
WORLD_RANK=$OMPI_COMM_WORLD_RANK
OMP_NUM_THREADS=1

python /shared/roshanin/training_results_v0.7/NVIDIA/benchmarks/maskrcnn/implementations/pytorch/tools/train_mlperf.py --config-file '/shared/roshanin/training_results_v0.7/NVIDIA/benchmarks/maskrcnn/implementations/pytorch/configs/e2e_mask_rcnn_R_50_FPN_1x.yaml' \
DTYPE 'float16' \
PATHS_CATALOG '/shared/roshanin/training_results_v0.7/NVIDIA/benchmarks/maskrcnn/implementations/pytorch/maskrcnn_benchmark/config/paths_catalog_dbcluster.py' \
DISABLE_REDUCED_LOGGING True \
SOLVER.BASE_LR ${BASE_LR} \
SOLVER.MAX_ITER ${MAX_ITER} \
SOLVER.WARMUP_FACTOR ${WARMUP_FACTOR} \
SOLVER.WARMUP_ITERS ${WARMUP_ITERS} \
SOLVER.WARMUP_METHOD mlperf_linear \
SOLVER.STEPS ${STEPS} \
SOLVER.IMS_PER_BATCH ${TRAIN_IMS_PER_BATCH} \
TEST.IMS_PER_BATCH ${TEST_IMS_PER_BATCH} \
MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN ${FPN_POST_NMS_TOP_N_TRAIN} \
NHWC True \
DATALOADER.NUM_WORKERS 1 \
SOLVER.CHECKPOINT_PERIOD 450 \
SAVE_CHECKPOINTS True \
OUTPUT_DIR '/shared/datasets/checkpoints_32_epoch17' \
PER_EPOCH_EVAL True
