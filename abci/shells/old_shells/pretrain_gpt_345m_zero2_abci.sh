#! /bin/bash
# Runs the "345M" parameter model

NODES=1
GPUS_PER_NODE=4

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=dataset/BookCorpusDataset_text_document
CHECKPOINT_PATH=checkpoints/gpt2/${JOB_ID}/
VOCAB_FILE=dataset/gpt2-vocab.json
MERGE_FILE=dataset/gpt2-merges.txt

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

# Parallelization config (1st stage)
TP=1
PP=1
GLOBAL_BATCH=64
MICRO_BATCH=8
ZERO_STAGE=2

# Deepspeed config (1st stage)
DS_CONFIG=ds_config.json
cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "steps_per_print": 1,

  "zero_optimization": {
    "stage": $ZERO_STAGE
  },

  "fp16": {
    "enabled": true,
    "initial_scale_power": 12
  },

  "wall_clock_breakdown" : true
}
EOT

ds_args=""
ds_args=" --deepspeed ${ds_args}"
ds_args=" --no-pipeline-parallel ${ds_args}" 
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"
ds_args=" --deepspeed-activation-checkpointing ${ds_args}"

export NCCL_DEBUG=warn 
#OUTPUT_DIR=ds_z${ZERO_STAGE}_nl${NLAYERS}_hs${HIDDEN}_gb${GLOBAL_BATCH}_mb${MICRO_BATCH}
#mkdir -p $OUTPUT_DIR

HOSTFILE_NAME=./hostfile_${JOB_ID}
cat $SGE_JOB_HOSTLIST > $HOSTFILE_NAME

gpt_options=" \
       --tensor-model-parallel-size ${TP} \
       --pipeline-model-parallel-size ${PP} \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size ${MICRO_BATCH}  \
       --global-batch-size ${GLOBAL_BATCH} \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 500 \
       --lr-decay-iters 320 \
       --save ${CHECKPOINT_PATH} \
       --load ${CHECKPOINT_PATH} \
       --data-path ${DATA_PATH} \
       --vocab-file ${VOCAB_FILE} \
       --merge-file ${MERGE_FILE} \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --checkpoint-activations \
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       ${ds_args} \
       --fp16"


deepspeed --num_nodes ${NODES} --num_gpus ${GPUS_PER_NODE} pretrain_gpt.py $@ ${gpt_options}