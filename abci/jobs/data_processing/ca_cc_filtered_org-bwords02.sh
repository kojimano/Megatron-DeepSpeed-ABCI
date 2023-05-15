#!/bin/bash
#$ -l rt_C.large=1
#$ -l h_rt=7:00:00
#$ -j y
#$ -cwd
#$ -o /bb/grandchallenge/gaf51090/job_outputs/ca_cc_filtered_org-bwords02.$JOB_ID

source /etc/profile.d/modules.sh
module load python/3.10/3.10.10 cuda/11.8/11.8.0 cudnn/8.6/8.6.0 nccl/2.16/2.16.2-1
module unload intel-mpi/2021.8
module load hpcx/2.12
source /home/acf15317dw/megatron-deepspeed/bin/activate
cd /home/acf15317dw/Megatron-DeepSpeed-ABCI

export OUTDIR=/bb/grandchallenge/gaf51090/datasets/cacc/binarized/sp_identity
mkdir -p $OUTDIR

python tools/preprocess_data.py \
        --input /bb/grandchallenge/gaf51090/datasets/cacc/processed/ca_cc_filtered_org-bwords02.jsonl \
        --output-prefix $OUTDIR/ca_cc_filtered_org-bwords02 \
        --vocab-file /bb/grandchallenge/gaf51090/tokenizer_new/spm_input_fall_replaced_all_identity_wodummyprefix_modified.model \
        --dataset-impl mmap \
        --tokenizer-type JapaneseSentencePiece \
        --workers 32 \
        --append-eod
