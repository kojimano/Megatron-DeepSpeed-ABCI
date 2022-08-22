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
#MODEL_SIZE=0.125
#NUM_LAYERS=12
#HIDDEN_SIZE=768
#NUM_ATTN_HEADS=12
#GLOBAL_BATCH_SIZE=256
#LR=6.0e-4
#MIN_LR=6.0e-5
#OUTPUT_BASEPATH=$DIR/output

## GPT-3 Medium 350M
# MODEL_SIZE=0.35
# NUM_LAYERS=24
# HIDDEN_SIZE=1024
# NUM_ATTN_HEADS=16
# GLOBAL_BATCH_SIZE=256
# LR=3.0e-4
# MIN_LR=3.0e-5
# OUTPUT_BASEPATH=$DIR/tensor_parallelism/350M/mem

## GPT-3 Large 760M
# MODEL_SIZE=0.76
# NUM_LAYERS=24
# HIDDEN_SIZE=1536
# NUM_ATTN_HEADS=16
# GLOBAL_BATCH_SIZE=256
# LR=2.5e-4
# MIN_LR=2.5e-5
# OUTPUT_BASEPATH=$DIR/output

## GPT-3 XL 1.3B
#MODEL_SIZE=1.3
#NUM_LAYERS=24
#HIDDEN_SIZE=2048
#NUM_ATTN_HEADS=16
#GLOBAL_BATCH_SIZE=640
#LR=2.0e-4
#MIN_LR=2.0e-5
#OUTPUT_BASEPATH=$DIR/pipelining/



## GPT-3 2.7B
#MODEL_SIZE=2.7
#NUM_LAYERS=32
#HIDDEN_SIZE=2560
#NUM_ATTN_HEADS=32
#GLOBAL_BATCH_SIZE=32
#LR=1.6e-4
#MIN_LR=1.6e-5
#OUTPUT_BASEPATH=$DIR/pipelining

## GPT-Neo 6B
#MODEL_SIZE=5.85
#NUM_LAYERS=28
#HIDDEN_SIZE=4096
#NUM_ATTN_HEADS=16
##GLOBAL_BATCH_SIZE=32
#LR=1.2e-4
#MIN_LR=1.2e-5
#OUTPUT_BASEPATH=$DIR/gpt_neo_6b

## GPT-3 6.7B
#MODEL_SIZE=6.7
#NUM_LAYERS=32
#HIDDEN_SIZE=4096
#NUM_ATTN_HEADS=32
#GLOBAL_BATCH_SIZE=32
#LR=1.2e-4
#MIN_LR=1.2e-5
#OUTPUT_BASEPATH=$DIR/gpt_6.7B

## GPT-3 13B
# MODEL_SIZE=13
# NUM_LAYERS=40
## HIDDEN_SIZE=5120
# NUM_ATTN_HEADS=40
# GLOBAL_BATCH_SIZE=1024
# LR=1.0e-4
# MIN_LR=1.0e-5

## GPT-3 175B
# MODEL_SIZE=175
# NUM_LAYERS=96
# HIDDEN_SIZE=12288
# NUM_ATTN_HEADS=96
# GLOBAL_BATCH_SIZE=1536
# LR=0.6e-4
# MIN_LR=0.6e-5
###############################################################################
### Training duration configs
## The main termination condition, original GPT-3 paper trains for 300B tokens
### For MoE model, we found sometimes training a bit more to 330B tokens helps
TRAIN_TOKENS=300000000000
# TRAIN_TOKENS=330000000000

## TRAIN_ITERS is another termination condition and also affect the number of 
## data samples to be indexed. Since we want to reach the TRAIN_TOKENS
## above, and techniques like curriculum learning has less token in some steps,
## so we just set this config large enough to make sure we have enough
## processed data and don't terminate by TRAIN_ITERS.
#TRAIN_ITERS=20

## Another termination condition in minutes. Set it large enough to avoid
## undesired early termination.
EXIT_DURATION=30000000
###############################################################################
### LR configs
## LR warmup and decay duration, this token-based config is preferable since
## no need to readjust when the batch size/seqlen is changed.
## Original GPT-3 paper uses 375M warmup tokens and 260B decay tokens.
## For MoE model, we found that setting the decay token to 300B helps.
WARMUP_TOKENS=375000000
# LR_DECAY_TOKENS=260000000000
LR_DECAY_TOKENS=300000000000
###############################################################################
### Parallelism configs
## Micro batch size per GPU
## Make sure that BATCH_SIZE <= GLOBAL_BATCH_SIZE*PP_SIZE*MP_SIZE/NUM_GPUS


## Model parallelism, 1 is no MP
## Currently MoE models have divergence issue when MP > 1.
#MP_SIZE=4

## Pipeline parallelism
## Currently we don't support PP for MoE. To disable PP, set PP_SIZE
## to 1 and use the "--no-pipeline-parallel" arg.
PP_SIZE=1

## Number of experts
# EP_SIZE=8

## ZERO STAGE (0 or 2)

##Optimizer offloading
CPU_OFFLOAD="false"


#NUM_NODES=4
#NUM_GPUS_PER_NODE=8
#NUM_GPUS=$((NUM_NODES * NUM_GPUS_PER_NODE))

#MODEL_SIZE=2.7
#NUM_LAYERS=1
#HIDDEN_SIZE=7168 
#HIDDEN_SIZE=3584 
#NUM_ATTN_HEADS=56

export NCCL_CROSS_NIC=1

NUM_LAYERS_LIST=( 20 )
HIDDEN_SIZES=( 5120 ) 
ATTN_HEADS=( 40 )
MODEL_NAME="6.7B"


TOTAL=${#HIDDEN_SIZES[@]}
LR=1.6e-4
MIN_LR=1.6e-5

# GLOBAL_BATCH_SIZE=1024
EXPERT_MODEL_PARALLEL="true"
DENSE="false"
WALL_CLOCK_BREAKDOWN="false"
DROP_DUPLICATES_BEFORE_GATING="false"
#USE_TUTEL="false"

NUM_NODES=16
NUM_GPUS_PER_NODE=8
NUM_GPUS=$((NUM_NODES * NUM_GPUS_PER_NODE))


EP_SIZE=$(( NUM_GPUS / 2 ))
OUTPUT_BASEPATH="$DIR/large_runs/${MODEL_NAME}_base/convergence"


ZERO_STAGE=1
#BATCH_SIZE=4	
GLOBAL_BATCH_SIZE=$(( 16 * NUM_GPUS ))
USE_TUTEL="true"
WORKERS="1"
#GAS=1 #128

#cd ~/DeepSpeed
#ASYNC_EXPERTS="true"


i=0
#if [ "${ASYNC_EXPERTS}" = "true" ]; then
#	git checkout async-experts
#else
#	git checkout moe-full-tp
#fi


while [ $i -lt $TOTAL ];
    do
    for BATCH_SIZE in 4
    do
    for MP_SIZE in 2
    do 
    HIDDEN_SIZE=${HIDDEN_SIZES[i]}
    NUM_ATTN_HEADS=${ATTN_HEADS[i]}
    NUM_LAYERS=${NUM_LAYERS_LIST[i]}
    BASE_MODEL_SIZE="$(( 12 * NUM_LAYERS * HIDDEN_SIZE * HIDDEN_SIZE / 1000000000 ))B"
    MIN_GBS=$(( BATCH_SIZE * NUM_GPUS / MP_SIZE )) 
    if [ ${MIN_GBS} -gt ${GLOBAL_BATCH_SIZE} ]; then
	  continue 1 
    else
    	GAS=$(( GLOBAL_BATCH_SIZE * MP_SIZE / BATCH_SIZE / NUM_GPUS  ))
    fi	
    #TRAIN_ITERS=$(( ${TRAIN_TOKENS} * 3 / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
    TRAIN_ITERS=3000
    if [ "${EXPERT_MODEL_PARALLEL}" = "true" ]; then
    	MAX_EP_PARALLEL_SIZE=$(( NUM_GPUS / MP_SIZE ))
	#EP_SIZE=$(( NUM_GPUS / MP_SIZE ))
    else
   	MAX_EP_PARALLEL_SIZE=${NUM_GPUS}
	#EP_SIZE=${NUM_GPUS}
    fi

    if [ "${DENSE}" = "true" ];then
	echo "Running a dense model..."
    	EP_SIZE=1
    fi
    echo Running with batch size ${GLOBAL_BATCH_SIZE} mbs ${BATCH_SIZE} and ${EP_SIZE} experts
    MODEL_SIZE="${NUM_LAYERS}_layers_${HIDDEN_SIZE}_hsize_${EP_SIZE}_MoE"

    if [ ${EP_SIZE} -gt ${MAX_EP_PARALLEL_SIZE} ]; then
        EP_PARALLEL_SIZE=${MAX_EP_PARALLEL_SIZE}
    else
        EP_PARALLEL_SIZE=${EP_SIZE}
    fi
    
   #if [[ $EP_SIZE -gt $NUM_GPUS ]]; then
    #    EP_PARALLEL_SIZE=$NUM_GPUS
    #else
    #    EP_PARALLEL_SIZE=$EP_SIZE
    #fi

    ## Original GPT-3 model always set min LR at 10% of max LR. For MoE model, we
    ## found that lower LR and min LR (than the base dense model) helps.
    ## For 1.3B MoE-128 model we used LR=1.2e-4 and MIN_LR=1.0e-6.
    ## For 350M MoE-128 model we used LR=2.0e-4 and MIN_LR=2.0e-6, but they are not
    ## heavily tuned.
    LR=1.2e-4
    MIN_LR=1.2e-5

    ## Coefficient for MoE loss. We find that 0.01 is a good value at least for
    ## 1.3B MoE-128 model
    MLC=0.01

    ## Below configs adjust the MoE expert token capacity limit during training and
    ## eval. To completely disable capacity limit, set MOE_DROP_TOKEN to false.
    ## Larger capacity factor or disabling capacity limit could improve training
    ## convergence, but will also reduce training throughput.
    MOE_TRAIN_CAP_FACTOR=1.0
    MOE_EVAL_CAP_FACTOR=1.0
    MOE_MIN_CAP=4
    MOE_DROP_TOKEN="true"
    # MOE_DROP_TOKEN="false"
    ###############################################################################
    ### Curriculum learning (CL) configs
    ## Enable/disable CL
    CL_ENABLED="false"
    ## Consult the tutorial https://www.deepspeed.ai/tutorials/curriculum-learning/
    ## for tuning the following configs
    CL_START_SEQLEN=80
    CL_AVG_SEQLEN=$(( (${CL_START_SEQLEN} + ${SEQ_LEN}) / 2 ))
    CL_TOKENS=60
    CL_TOKENS=$((${CL_TOKENS} * 1000000000))
    CL_STEP=$(( ${CL_TOKENS} / (${GLOBAL_BATCH_SIZE} * ${CL_AVG_SEQLEN}) ))
    ###############################################################################
    ### Misc configs
    LOG_INTERVAL=1
    EVAL_ITERS=10
    EVAL_INTERVAL=10
    SAVE_INTERVAL=10000

    ## Standard deviation for weight initialization
    ## We used 0.014 for 350M/1.3B dense/MoE models, and used 0.01 for 6.7B
    ## dense model. Usually larger model needs lower std.
    INIT_STD=0.014
    # INIT_STD=0.01

    ## Activation checkpointing saves GPU memory, but reduces training speed
    ACTIVATION_CHECKPOINT="true"
    # ACTIVATION_CHECKPOINT="false"
    ###############################################################################
    ### Output and data configs
    current_time=$(date "+%Y.%m.%d-%H.%M.%S")
    host="${HOSTNAME}"
    export MAX_SPIKE_GB=4

    NAME="gpt-${MODEL_SIZE}-lr-${LR}-minlr-${MIN_LR}-bs-${GLOBAL_BATCH_SIZE}-mbs-${BATCH_SIZE}-gpus-${NUM_GPUS}-mp-${MP_SIZE}-pp-${PP_SIZE}-tutel-${USE_TUTEL}-drop_before_gating-${DROP_DUPLICATES_BEFORE_GATING}-workers-${WORKERS}"
    if [ "${EXPERT_MODEL_PARALLEL}" = "true" ]; then
    	NAME="${NAME}-full-mp"
    fi
    if [[ $EP_SIZE -gt 1 ]]; then
        NAME="${NAME}-ep-${EP_SIZE}-mlc-${MLC}-cap-${MOE_TRAIN_CAP_FACTOR}-drop-${MOE_DROP_TOKEN}-maxspikegb-${MAX_SPIKE_GB}"
    fi
    if [ "${CL_ENABLED}" = "true" ]; then
        NAME="${NAME}-cl-${CL_START_SEQLEN}-${CL_STEP}"
    fi
    if [ "${CPU_OFFLOAD}" = "true" ]; then
        NAME="${NAME}-offload"
        echo "Running with ZeRO-Offload"
    fi
   

    mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
    mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
    mkdir -p "${OUTPUT_BASEPATH}/log/"
    TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${NAME}_${host}_${current_time}"
    mkdir -p ${TENSORBOARD_DIR} 
    ## Note that for MoE model with billion-scale base model, the checkpoint can be
    ## as large as TB-scale which normal NFS cannot handle efficiently.
    CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${NAME}"

    # USE_INTERNAL_DATA="true"
    USE_INTERNAL_DATA="false"

    if [ "${USE_INTERNAL_DATA}" = "true" ]; then
        ## The internal data is only accessible within Microsoft
        ## For cluster Azure-EastUS-V100-32GB-4, Azure-WestUS3-A100
        # BASE_DATA_PATH=/vc_data/Megatron-LM/data
        # DATA_HOME="/vc_data/pile-cc1-cc2-shuf"
        ## For cluster Lab-RR1-V100
        BASE_DATA_PATH=/data/Megatron-LM/data
        DATA_HOME="/turing-ssd/users/conglli/data/pile-cc1-cc2-shuf"
        ## For cluster Azure-CentralUS-A100
        # BASE_DATA_PATH=/data/Megatron-LM/data
        # DATA_HOME=/vc_data_1/users/amawa/blended

        VOCAB_PATH=${BASE_DATA_PATH}/gpt2-vocab.json
        MERGE_PATH=${BASE_DATA_PATH}/gpt2-merges.txt
        ARX="${DATA_HOME}/ArXiv_ftfy_cleaned_id_shuf_text_document"
        BC2="${DATA_HOME}/BookCorpus2_ftfy_cleaned_id_shuf_text_document"
        B3="${DATA_HOME}/Books3_ftfy_cleaned_id_shuf_text_document"
        CC2020="${DATA_HOME}/CC-2020-50_id_cleaned_shuf_text_document"
        CC2021="${DATA_HOME}/CC-2021-04_id_cleaned_shuf_text_document"
        GIT="${DATA_HOME}/Github_ftfy_id_shuf_text_document"
        GUT="${DATA_HOME}/Gutenberg_PG-19_ftfy_cleaned_id_cleaned_shuf_text_document"
        NIH="${DATA_HOME}/NIH_ExPorter_ftfy_id_shuf_text_document"
        OWT2="${DATA_HOME}/OpenWebText2_ftfy_cleaned_id_shuf_text_document"
        PCC="${DATA_HOME}/Pile-CC_id_cleaned_shuf_text_document"
        PM="${DATA_HOME}/PubMed_Abstracts_ftfy_id_shuf_text_document"
        RN="${DATA_HOME}/rn_dedup_shuf_cleaned_0.7_cleaned_shuf_text_document"
        SE="${DATA_HOME}/StackExchange_ftfy_id_shuf_text_document"
        ST="${DATA_HOME}/stories_dedup0.7_shuf_cleaned_shuf_text_document"
        WIK="${DATA_HOME}/Wikipedia_en_ftfy_id_shuf_text_document"
        DATA_BLEND="0.14336 ${B3} 0.08962 ${RN} 0.19336 ${OWT2} 0.05689 ${SE} \
        0.00859 ${ST} 0.02897 ${PM} 0.04771 ${WIK} 0.00873 ${GUT} 0.01007 ${BC2} \
        0.00208 ${NIH} 0.13017 ${CC2020} 0.09446 ${PCC} 0.15652 ${CC2021} \
        0.01359 ${ARX} 0.01588 ${GIT}"
    else
	DATA_PATH=/data/users/ssingh37
        VOCAB_PATH=${DATA_PATH}/gpt2-vocab.json
        MERGE_PATH=${DATA_PATH}/gpt2-merges.txt
        # Public the Pile dataset, can be downloaded at https://mystic.the-eye.eu/public/AI/pile_neox/
        DATA_BLEND=${DATA_PATH}/BookCorpusDataset_text_document
    fi
    ###############################################################################
    data_options=" \
            --vocab-file ${VOCAB_PATH} \
            --merge-file ${MERGE_PATH} \
            --data-path ${DATA_BLEND} \
            --data-impl mmap"
            
    megatron_options=" \
            --override-lr-scheduler \
            --adam-beta1 0.9 \
            --adam-beta2 0.95 \
            --tensor-model-parallel-size ${MP_SIZE} \
            --moe-expert-parallel-size ${EP_PARALLEL_SIZE} \
            --num-experts ${EP_SIZE} \
            --moe-loss-coeff ${MLC} \
            --moe-train-capacity-factor ${MOE_TRAIN_CAP_FACTOR} \
            --moe-eval-capacity-factor ${MOE_EVAL_CAP_FACTOR} \
            --moe-min-capacity ${MOE_MIN_CAP} \
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
            --train-iters ${TRAIN_ITERS} \
            --lr ${LR} \
            --min-lr ${MIN_LR} \
            --lr-decay-style cosine \
            --split 98,2,0 \
            --log-interval ${LOG_INTERVAL} \
            --eval-interval ${EVAL_INTERVAL} \
            --eval-iters ${EVAL_ITERS} \
            --save-interval ${SAVE_INTERVAL} \
            --weight-decay 0.1 \
            --clip-grad 1.0 \
            --hysteresis 2 \
            --num-workers ${WORKERS} \
            --fp16 \
            --tensorboard-queue-size 1 \
            --log-timers-to-tensorboard \
            --log-batch-size-to-tensorboard \
            --log-validation-ppl-to-tensorboard \
            --tensorboard-dir ${TENSORBOARD_DIR}"
    		# --load ${CHECKPOINT_PATH} \
    		# --save ${CHECKPOINT_PATH} \
    		# ^these have been removed



    if [ "${PP_SIZE}" = "1" ]; then
    	megatron_options="${megatron_options} \
            --no-pipeline-parallel"
    fi

    if [ "${DROP_DUPLICATES_BEFORE_GATING}" = "true" ]; then
    	megatron_options="${megatron_options} \
            --drop-duplicates-before-gating"
    fi

    if [ "${USE_TUTEL}" = "true" ]; then
    	megatron_options="${megatron_options} \
            --use-tutel"
    fi

    if [ "${EXPERT_MODEL_PARALLEL}" = "true" ]; then
    	megatron_options="${megatron_options} \
            --enable-expert-tensor-parallelism"
    fi

    if [ "${ACTIVATION_CHECKPOINT}" = "true" ]; then
    megatron_options="${megatron_options} \
            --checkpoint-activations"
    fi

    if [ "${CPU_OFFLOAD}" = "true" ]; then
    megatron_options="${megatron_options} \
            --cpu-optimizer"
    fi

    if [[ $EP_SIZE -gt 1 ]]; then
    megatron_options="${megatron_options} \
            --create-moe-param-group"
    fi

    if [ "${MOE_DROP_TOKEN}" = "false" ]; then
    megatron_options="${megatron_options} \
            --disable-moe-token-dropping"
    fi

    if [[ ${ZERO_STAGE} -ne 3 ]]; then
        template_json="ds_config_gpt_TEMPLATE.json"
        echo "Running Zero Stage - ${ZERO_STAGE}"
    else 
        template_json="ds_config_gpt_Zero3_TEMPLATE.json"
        echo "Running Zero Stage - ${ZERO_STAGE}"
    fi
    config_json="ds_config_gpt_${NAME}.json"

    sed "s/CONFIG_BATCH_SIZE/${GLOBAL_BATCH_SIZE}/" ${template_json} \
        | sed "s/CONFIG_MBSIZE/${BATCH_SIZE}/" \
        | sed "s/LOG_INTERVAL/${LOG_INTERVAL}/" \
        | sed "s/ZERO_STAGE/${ZERO_STAGE}/" \
        | sed "s/PRESCALE_GRAD/false/" \
        | sed "s/CONFIG_FP16_ENABLED/true/" \
        | sed "s/CONFIG_BF16_ENABLED/false/" \
        | sed "s/CONFIG_CL_ENABLED/${CL_ENABLED}/" \
        | sed "s/CONFIG_CL_MIN/${CL_START_SEQLEN}/" \
        | sed "s/CONFIG_CL_MAX/${SEQ_LEN}/" \
        | sed "s/CONFIG_CL_DURATION/${CL_STEP}/" \
        | sed "s/WALL_CLOCK_BREAKDOWN/${WALL_CLOCK_BREAKDOWN}/" \
        | sed "s/CPU_OFFLOAD/${CPU_OFFLOAD}/" \
        > ${config_json}
   
    deepspeed_options=" \
                --deepspeed \
                --deepspeed_config ${config_json} \
                --pipeline-model-parallel-size ${PP_SIZE}"


    if [ "${ACTIVATION_CHECKPOINT}" = "true" ]; then
    deepspeed_options="${deepspeed_options} \
            --deepspeed-activation-checkpointing"
    fi
    
    export AZUREML_PARAMETER_ITPJOB_NAME="scaling runs"
    export NCCL_DEBUG=VERSION
    
    # ds_ssh pkill -9 python
    # ds_ssh pkill -9 python
    # ds_ssh pkill -9 python
    # ds_ssh pkill -9 python
    # ds_ssh pkill -9 python

    run_cmd="deepspeed --num_nodes ${NUM_NODES} --num_gpus ${NUM_GPUS_PER_NODE} ${DIR}/../../pretrain_gpt.py ${megatron_options} ${data_options} ${deepspeed_options} | tee ${OUTPUT_BASEPATH}/log/${NAME}_${host}_${current_time}.log"
    echo ${run_cmd}
    eval ${run_cmd}
    set +x
    # ds_ssh pkill -9 python
    # ds_ssh pkill -9 python
    # ds_ssh pkill -9 python
    # ds_ssh pkill -9 python
done
done
let i++
done
