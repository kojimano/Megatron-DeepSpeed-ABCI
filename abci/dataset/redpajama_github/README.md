# Redpajama Code Data Preparation

## Table of Contents
TBD

## Procedures

### 1. Get computes (ABCI) and install packages
```bash
# 1.1 Get computes
qrsh -g gaf51090 -l rt_C.small=1 -l h_rt=2:00:00 

# 1.2 Activate your virtual environment
source XXX/bin/activate
```

### 2. Tokenize and binarize data into the megatron-deepspeed format
### 2.1. Generate debug jsonl by splitting large jsonl:
```bash
head -n 500000 /bb/grandchallenge/gaf51090/datasets/redpajama_github/merged/merged.jsonl > /bb/grandchallenge/gaf51090/datasets/redpajama_github/merged/debug.jsonl
```

### 2.2. Tokenize and binarize data:

**Sentencepiece (ours):**

Tokenization and binarization for `Debug` splits
```bash
export OUTDIR=/bb/grandchallenge/gaf51090/datasets/redpajama_github/binarized/sentencepiece_ver1
mkdir -p $OUTDIR
python tools/preprocess_data.py \
        --input /bb/grandchallenge/gaf51090/datasets/redpajama_github/merged/debug.jsonl \
        --output-prefix $OUTDIR/redpajama_github_debug \
        --vocab-file /bb/grandchallenge/gaf51090/datasets/tokenization_replaced/spm_input_fall_replaced_all.model \
        --dataset-impl mmap \
        --tokenizer-type JapaneseSentencePiece \
        --workers 32 \
        --append-eod

export OUTDIR=/bb/grandchallenge/gaf51090/datasets/redpajama_github/binarized/sp_nmt_nfkc_with_ws_tab
mkdir -p $OUTDIR
python tools/preprocess_data.py \
        --input /bb/grandchallenge/gaf51090/datasets/redpajama_github/merged/debug.jsonl \
        --output-prefix $OUTDIR/redpajama_github_debug \
        --vocab-file /bb/grandchallenge/gaf51090/tokenizer_new/spm_input_fall_replaced_all_wodummyprefix_modified.model \
        --dataset-impl mmap \
        --tokenizer-type JapaneseSentencePiece \
        --workers 32 \
        --append-eod        

export OUTDIR=/bb/grandchallenge/gaf51090/datasets/redpajama_github/binarized/sp_identity
mkdir -p $OUTDIR
python tools/preprocess_data.py \
        --input /bb/grandchallenge/gaf51090/datasets/redpajama_github/merged/debug.jsonl \
        --output-prefix $OUTDIR/redpajama_github_debug \
        --vocab-file /bb/grandchallenge/gaf51090/tokenizer_new/spm_input_fall_replaced_all_identity_wodummyprefix_modified.model \
        --dataset-impl mmap \
        --tokenizer-type JapaneseSentencePiece \
        --workers 32 \
        --append-eod

```

Tokenize and binarization
```bash
```


## Dataset Statistics

Here's the formatted table:

### Number of Tokens
| Split | SP (nmt_nfkc) | SP (nmt_nfkc) \w ws & tab | SP (identity) |
| -------- | -------------------- | ---------------------- |---------------- |
| Debug (500k docs) | 1,149,912,083 | 1,571,851,576 | 1,571,993,571  |
| redpajama_github00 | -          | -            | -     |
| redpajama_github01 | -          | -            | -     |
| redpajama_github02 | -          | -            | -     |
| redpajama_github03 | -          | -            | -     |


### Data Paths

- Pathes under `/bb/grandchallenge/gaf51090/datasets/redpajama_github`


