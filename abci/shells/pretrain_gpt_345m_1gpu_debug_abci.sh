#! /bin/bash

## Multinode config
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0 

# Data path
DATA_PATH='
       10 /bb/grandchallenge/gaf51090/datasets/aozora_books/binarized/abeja/aozora_books_text_document
       45 /bb/grandchallenge/gaf51090/datasets/wikipedia/binarized/abeja/en_wiki_text_document
       45 /bb/grandchallenge/gaf51090/datasets/wikipedia/binarized/abeja/ja_wiki_text_document
'

# Model size
NUM_LAYERS=24
HIDDEN_SIZE=1024
NUM_ATTENTION_HEADS=16

# Other experimental parameters
CHECKPOINT_PATH=/bb/grandchallenge/gaf51090/checkpoints/345m_1gpu_debug
TENSORBOARD_PATH=/bb/grandchallenge/gaf51090/logs/345m_1gpu_debug
VOCAB_FILE=''

python pretrain_gpt.py \
       --num-layers $NUM_LAYERS \
       --hidden-size $HIDDEN_SIZE \
       --num-attention-heads $NUM_ATTENTION_HEADS \
       --micro-batch-size 1 \
       --global-batch-size 80 \
       --seq-length 2048 \
       --max-position-embeddings 2048 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --tensorboard-dir $TENSORBOARD_PATH \
       --train-iters 200 \
       --data-path $DATA_PATH \
       --tokenizer-type AbejaJapaneseGPT2Tokenizer \
       --data-impl mmap \
       --split 949,50,1 \
       --lr 0.0001 \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --hidden-dropout 0 \
       --attention-dropout 0 \
       --weight-decay 0.1 \
       --log-interval 10 \
       --save-interval 100 \
       --eval-interval 100 \
       --eval-iters 10 \
       --fp16 \
       --checkpoint-activations 


rm $HOSTFILE_NAME

