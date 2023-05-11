CHECKPOINT_PATH=/bb/grandchallenge/gaf51090/checkpoints/japanese_gptneox/39806947
VOCAB_FILE=/bb/grandchallenge/gaf51090/tokenizer/abeja_japanese_sp.model

MAX_OUTPUT_SEQUENCE_LENGTH=1024
TEMPERATURE=1.0
TOP_P=0.9
NUMBER_OF_SAMPLES=2
OUTPUT_FILE=samples.json

python tools/generate_samples_gpt.py \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 1 \
       --global-batch-size 4 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 10000 \
       --lr-decay-iters 320 \
       --vocab-file $VOCAB_FILE \
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
       --save-interval 1000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --load $CHECKPOINT_PATH \
       --out-seq-length $MAX_OUTPUT_SEQUENCE_LENGTH \
       --temperature $TEMPERATURE \
       --genfile $OUTPUT_FILE \
       --num-samples $NUMBER_OF_SAMPLES \
       --top_p $TOP_P \
       --recompute
