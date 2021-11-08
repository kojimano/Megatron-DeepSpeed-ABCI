#!/bin/bash

CHECKPOINT_PATH=/tmp/shaden/textgen-mp2-pp2-zero1-seq1024-24l-1024h-ah32/global_step4000
VOCAB_FILE=/data/Megatron-LM/data/gpt2-vocab.json
MERGE_FILE=/data/Megatron-LM/data/gpt2-merges.txt

MP=2
PP=1
GPUS=$(( $MP * $PP ))

MBSIZE=1
ZERO_STAGE=1

# Generate DS config
CONFIG_JSON=$(pwd)/ds_config_gen.json
cat <<EOT > $CONFIG_JSON
{
  "train_batch_size" : 1,
  "train_micro_batch_size_per_gpu": $MBSIZE,
  "steps_per_print": 1,

  "zero_optimization": {
    "stage": $ZERO_STAGE
  },

  "gradient_clipping": 1.0,

  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 500,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 12
  },

  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.001
    }
  },

  "wall_clock_breakdown" : true
}
EOT

deepspeed --num_gpus=$GPUS tools/generate_samples_gpt.py \
       --deepspeed \
       --deepspeed_config ${CONFIG_JSON} \
       --tensor-model-parallel-size ${MP} \
       --pipeline-model-parallel-size ${PP} \
       --num-layers 24 \
       --hidden-size 1024 \
       --load $CHECKPOINT_PATH \
       --num-attention-heads 32 \
       --max-position-embeddings 1024 \
       --tokenizer-type GPT2BPETokenizer \
       --fp16 \
       --micro-batch-size 1 \
       --seq-length 1024 \
       --out-seq-length 1024 \
       --temperature 1.0 \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGE_FILE \
       --sample-input-file hello.txt \
       --sample-output-file out-pp${PP}.txt \
       --genfile unconditional_samples.json \
       --num-samples 0 \
       --top_p 0.9 \
       #--recompute
