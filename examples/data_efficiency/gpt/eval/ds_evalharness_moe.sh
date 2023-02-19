## CAUTION: first read Megatron-DeepSpeed/blob/main/examples/MoE/readme_evalharness.md
## and follow the steps of installation/data downloading.

checkpoint_path=$1
hostname_and_rank=$2
master_port=$3
ep_size=$4
batch_size=$5
num_fewshot=$6
tasks=$7

## No need to use the exact training config json, just use this dummy is fine
config_path=ds_config_eval_dummy.json

username=$(whoami)
result_path="/blob/users/${username}/project/data_efficient_gpt/eval_results_${num_fewshot}shot"

mp_size=1
pp_size=1
no_pp="true"

vocab_file="gpt2-vocab.json"
if [ ! -f "$vocab_file" ]; then
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
fi
merge_file="gpt2-merges.txt"
if [ ! -f "$merge_file" ]; then
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
fi

export HF_DATASETS_OFFLINE=1

dir2=$(dirname "$checkpoint_path")
dirname=$(basename "$dir2")/$(basename "$checkpoint_path")
result_path="${result_path}/${dirname}"
mkdir -p $result_path
result_file="${result_path}/${tasks}_${num_fewshot}shot.json"

# Dummy arguments to make megatron happy. No need to configure them.
# The reason we don't need to configure them and many other arguments is
# because the eval framework will read the arguments from checkpoint file.
megatron_required_args="\
    --num-layers -1 \
    --hidden-size -1 \
    --num-attention-heads -1 \
    --seq-length -1 \
    --max-position-embeddings -1
"

command="../../../../tasks/eval_harness/evaluate.py \
    --load ${checkpoint_path} \
    --tensor-model-parallel-size ${mp_size} \
    --pipeline-model-parallel-size ${pp_size} \
    --moe-expert-parallel-size ${ep_size} \
    --vocab-file ${vocab_file} \
    --merge-file ${merge_file} \
    --micro-batch-size ${batch_size} \
    --no-load-optim \
    --no-load-rng \
    --inference \
    --disable-moe-token-dropping \
    --adaptive_seq_len \
    --eval_fp32 \
    --num_fewshot ${num_fewshot} \
    --task_list ${tasks} \
    --results_path ${result_file} \
    --deepspeed \
    --deepspeed_config ${config_path} \
    ${megatron_required_args} \
    "

if [[ "${no_pp}" = "true" ]]; then
command="${command} \
    --no-pipeline-parallel"
fi

launcher="deepspeed --include=${hostname_and_rank} --master_port=${master_port}"
$launcher $command &> "${result_path}/${tasks}_${num_fewshot}shot.log"