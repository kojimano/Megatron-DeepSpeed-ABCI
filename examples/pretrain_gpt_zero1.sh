#! /bin/bash

# Runs the "10B" parameter model

GPUS_PER_NODE=4
## Change for multinode config
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))
NNODES=$NHOSTS
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
echo WORLD_SIZE $WORLD_SIZE

DATA_PATH=dataset/BookCorpusDataset_text_document
CHECKPOINT_PATH=checkpoints/gpt2/${JOB_ID}/
VOCAB_FILE=dataset/gpt2-vocab.json
MERGE_FILE=dataset/gpt2-merges.txt

HOSTFILE_NAME=./hosts/hostfile_${JOB_ID}
cat $SGE_JOB_HOSTLIST > $HOSTFILE_NAME
#DEEPSPEED_HOST_NAME=./hosts/hostfile_deepspeed_${JOB_ID}
#python make_deepspeed_hostfile.py --input-name $HOSTFILE_NAME --output-name $DEEPSPEED_HOST_NAME --slot $GPUS_PER_NODE

GLOBAL_BATCH=512
MICRO_BATCH=1
# ZERO_STAGE=1
# DS_CONFIG=config/ds_zero_stage_1_${JOB_ID}.config
# cat <<EOT > $DS_CONFIG
# {
#   "train_batch_size" : $GLOBAL_BATCH,
#   "train_micro_batch_size_per_gpu": $MICRO_BATCH,
#   "steps_per_print": 1,
# 
#   "zero_optimization": {
#     "stage": $ZERO_STAGE,
#     "reduce_bucket_size": 5e8
#   },
# 
#   "fp16": {
#     "enabled": true,
#     "initial_scale_power": 12
#   },
# 
#   "wall_clock_breakdown" : true
# }
# EOT

# ZERO_STAGE=2
# DS_CONFIG=config/ds_zero_stage_${ZERO_STAGE}_${JOB_ID}.config
# cat <<EOT > $DS_CONFIG
# {
#   "train_batch_size" : $GLOBAL_BATCH,
#   "train_micro_batch_size_per_gpu": $MICRO_BATCH,
#   "steps_per_print": 2,
# 
#   "zero_optimization": {
#     "stage": $ZERO_STAGE,
#     "overlap_comm": true,
#     "reduce_scatter": true,
#     "reduce_bucket_size": 5e8,
#     "allgather_bucket_size": 5e8
#   },
# 
#   "fp16": {
#     "enabled": true,
#     "initial_scale_power": 12
#   },
# 
#   "wall_clock_breakdown" : true
# }
# EOT

ZERO_STAGE=3
DS_CONFIG=config/ds_zero_stage_${ZERO_STAGE}_${JOB_ID}.config
cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "steps_per_print": 2,

  "zero_optimization": {
    "stage": 3,
    "contiguous_gradients": true,
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_prefetch_bucket_size": 1e7,
    "stage3_param_persistence_threshold": 1e5,
    "reduce_bucket_size": 1e7,
    "sub_group_size": 1e9,
    "offload_optimizer": {
        "device": "cpu"
    },
    "offload_param": {
        "device": "cpu"
    }
  },

  "fp16": {
    "enabled": true,
    "initial_scale_power": 12
  },

  "wall_clock_breakdown" : true
}
EOT

STAGE3_ARGS="--deepspeed --no-pipeline-parallel --zero-stage $ZERO_STAGE --use-pin-memory --zero-contigious-gradients --zero-reduce-bucket-size 10000000 --deepspeed-activation-checkpointing "
ZERO_INFINIY="--remote-device cpu --cpu-optimizer"

#deepspeed --num_nodes $NNODES --num_gpus $GPUS_PER_NODE --hostfile $DEEPSPEED_HOST_NAME --launcher \
mpirun -np $WORLD_SIZE -npernode $GPUS_PER_NODE --hostfile $HOSTFILE_NAME python \
       pretrain_gpt.py \
       $STAGE3_ARGS \
       $ZERO_INFINIY \
       --num-layers 24 \
       --hidden-size 4096 \
       --num-attention-heads 64 \
       --micro-batch-size $MICRO_BATCH \
       --global-batch-size $GLOBAL_BATCH \
       --deepspeed_config $DS_CONFIG \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 500 \
       --lr-decay-iters 3200 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGE_FILE \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16 \
       --use-mpi

