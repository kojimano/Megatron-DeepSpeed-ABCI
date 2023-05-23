GPUS_PER_NODE=4
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))
NNODES=$NHOSTS
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
echo WORLD_SIZE $WORLD_SIZE
HOSTFILE_NAME=./hostfile_${JOB_ID}
cat $SGE_JOB_HOSTLIST > $HOSTFILE_NAME

VOCAB_FILE='/bb/grandchallenge/gaf51090/tokenizer_new/spm_input_fall_replaced_all_identity_wodummyprefix_modified.model'

MAX_OUTPUT_SEQUENCE_LENGTH=48
TEMPERATURE=1.0
TOP_P=0.9
NUMBER_OF_SAMPLES=0
OUTPUT_FILE=samples_pretrain_gpt_13b_failure_17550.json

NUM_LAYERS=40
HIDDEN_SIZE=5120
NUM_ATTENTION_HEADS=40


mpirun -np $WORLD_SIZE -npernode $GPUS_PER_NODE --hostfile $HOSTFILE_NAME python tools/generate_samples_gpt.py \
       --tensor-model-parallel-size 4 \
       --pipeline-model-parallel-size 8 \
       --num-layers $NUM_LAYERS \
       --hidden-size $HIDDEN_SIZE \
       --num-attention-heads $NUM_ATTENTION_HEADS \
       --micro-batch-size 2 \
       --global-batch-size 2 \
       --seq-length 2048 \
       --max-position-embeddings 2048 \
       --load /bb/grandchallenge/gaf51090/checkpoints/13b_544gpu_commonspace_start_relu\
       --tokenizer-type JapaneseSentencePiece \
       --vocab-file $VOCAB_FILE \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 3e-6 \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --lr-decay-style cosine \
       --min-lr 3e-6 \
       --weight-decay 1e-2 \
       --clip-grad 0.05 \
       --init-method-std 0.00884 \
       --hidden-dropout 0 \
       --attention-dropout 0.1 \
       --weight-decay 0.1 \
       --log-interval 1 \
       --save-interval 250 \
       --eval-interval 10000 \
       --eval-iters 10 \
       --seed 1 \
       --out-seq-length $MAX_OUTPUT_SEQUENCE_LENGTH \
       --temperature $TEMPERATURE \
       --genfile $OUTPUT_FILE \
       --num-samples $NUMBER_OF_SAMPLES \
       --top_p $TOP_P \
       --fp16 \
       --no-bias-gelu-fusion \
       --use-relu \
       --no-load-lr-state  \
       --no-scatter-gather-tensors-in-pipeline \
       --use-mpi \
       #--recompute \

