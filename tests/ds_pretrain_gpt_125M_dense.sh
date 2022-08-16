#!/bin/bash
DIR=`pwd`
###############################################################################
### Main configs
## GPT-3 models use 2K sequence length/context window
SEQ_LEN=2048

### The "GPT-3 XXX" below are configs from GPT-3 paper
### https://arxiv.org/abs/2005.14165, choose based on
### your desired model size or build your own configs

## GPT-3 Small 125M
MODEL_SIZE=0.125
NUM_LAYERS=12
HIDDEN_SIZE=768
NUM_ATTN_HEADS=12
GLOBAL_BATCH_SIZE=256
LR=6.0e-4
MIN_LR=6.0e-5
###############################################################################
### Training duration configs
## The main termination condition, original GPT-3 paper trains for 300B tokens
NUM_STEP=20
TRAIN_TOKENS=$(( ${GLOBAL_BATCH_SIZE} * ${SEQ_LEN} * ${NUM_STEP}))
TRAIN_SAMPLES=$(( 20 * ${GLOBAL_BATCH_SIZE} ))

## Another termination condition in minutes. Set it large enough to avoid
## undesired early termination.
EXIT_DURATION=30000000
###############################################################################
### LR configs
## LR warmup and decay duration, this token-based config is preferable since
## no need to readjust when the batch size/seqlen is changed.
## Original GPT-3 paper uses 375M warmup tokens and 260B decay tokens.
WARMUP_TOKENS=375000000
LR_DECAY_TOKENS=260000000000
###############################################################################
### Parallelism configs
## Micro batch size per GPU
## Make sure that BATCH_SIZE <= GLOBAL_BATCH_SIZE*PP_SIZE*MP_SIZE/NUM_GPUS
BATCH_SIZE=4

## Model parallelism, 1 is no MP
MP_SIZE=1

## Pipeline parallelism. To disable PP, set PP_SIZE to 1 and NO_PP to true.
PP_SIZE=1
NO_PP="true"

## ZeRO stage
ZERO_STAGE=0

## Total number of GPUs
NUM_GPUS=2
NUM_GPU_PER_NODE=2
NUM_NODE=1
###############################################################################
### Misc configs
LOG_INTERVAL=1
EVAL_ITERS=1
EVAL_INTERVAL=2

## Standard deviation for weight initialization. Usually larger model needs
## lower std. We used a heuristic equation of sqrt(1/3/HIDDEN_SIZE) from the
## MT-NLG 530B work (https://arxiv.org/pdf/2201.11990.pdf)
INIT_STD=0.02
###############################################################################
### Output and data configs
VOCAB_PATH="gpt2-vocab.json"
if [ ! -f "$VOCAB_PATH" ]; then
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
fi
MERGE_PATH="gpt2-merges.txt"
if [ ! -f "$MERGE_PATH" ]; then
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
fi

DATA_PATH="BookCorpusDataset_text_document"
if [ ! -f "BookCorpusDataset_text_document.bin" ]; then
    wget https://the-eye.eu/public/AI/pile_neox/data/BookCorpusDataset_text_document.bin
fi
if [ ! -f "BookCorpusDataset_text_document.idx" ]; then
    wget https://the-eye.eu/public/AI/pile_neox/data/BookCorpusDataset_text_document.idx
fi

CHECKPOINT_PATH="checkpoint_dense"
###############################################################################
data_options=" \
         --vocab-file ${VOCAB_PATH} \
         --merge-file ${MERGE_PATH} \
         --data-path ${DATA_PATH} \
         --data-impl mmap"
        
megatron_options=" \
        --override-lr-scheduler \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --tensor-model-parallel-size ${MP_SIZE} \
        --init-method-std ${INIT_STD} \
        --lr-decay-tokens ${LR_DECAY_TOKENS} \
        --lr-warmup-tokens ${WARMUP_TOKENS} \
        --micro-batch-size ${BATCH_SIZE} \
        --exit-duration-in-mins ${EXIT_DURATION} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${SEQ_LEN} \
        --train-tokens ${TRAIN_TOKENS} \
        --train-samples ${TRAIN_SAMPLES} \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
        --split 98,2,0 \
        --log-interval ${LOG_INTERVAL} \
        --eval-interval ${EVAL_INTERVAL} \
        --eval-iters ${EVAL_ITERS} \
        --save-interval 10000 \
        --weight-decay 0.1 \
        --clip-grad 1.0 \
        --hysteresis 2 \
        --num-workers 0 \
        --fp16 \
        --load ${CHECKPOINT_PATH} \
        --save ${CHECKPOINT_PATH}"

config_json="ds_config_gpt.json"

deepspeed_options=" \
            --deepspeed \
            --deepspeed_config ${config_json} \
            --zero-stage ${ZERO_STAGE} \
            --pipeline-model-parallel-size ${PP_SIZE}"

if [[ "${NO_PP}" = "true" ]]; then
deepspeed_options="${deepspeed_options} \
        --no-pipeline-parallel"
fi

run_cmd="deepspeed --num_nodes $NUM_NODE --num_gpus $NUM_GPU_PER_NODE ${DIR}/../pretrain_gpt.py ${megatron_options} ${data_options} ${deepspeed_options}"
echo ${run_cmd}
eval ${run_cmd}
set +x
