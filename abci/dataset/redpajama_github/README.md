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
```bash
export OUTDIR=/bb/grandchallenge/gaf51090/datasets/redpajama_github/binarized/sentencepiece_ver1
mkdir -p $OUTDIR
```

Tokenize and binarize:
```bash
python tools/preprocess_data.py \
        --input /bb/grandchallenge/gaf51090/datasets/redpajama_github/merged/debug.jsonl \
        --output-prefix $OUTDIR/redpajama_github_debug \
        --vocab-file /bb/grandchallenge/gaf51090/datasets/tokenization_replaced/spm_input_fall_replaced_all.model \
        --dataset-impl mmap \
        --tokenizer-type JapaneseSentencePiece \
        --workers 32 \
        --append-eod

python tools/preprocess_data.py \
        --input /bb/grandchallenge/gaf51090/datasets/redpajama_github/merged/merged.jsonl \
        --output-prefix $OUTDIR/redpajama_github \
        --vocab-file /bb/grandchallenge/gaf51090/datasets/tokenization_replaced/spm_input_fall_replaced_all.model \
        --dataset-impl mmap \
        --tokenizer-type JapaneseSentencePiece \
        --workers 1 \
        --append-eod

```

## Dataset Statistics

Here's the formatted table:

### Basic Statistics

- Processing time calculated using `rt_C.small=1`
- (†) uses `login-node`

| # Extracted Blogs | # Discarded Blogs | Jsonl Size | # Tokens (Ours Ver.1) | # Tokens (GPT-2) | # Tokens (Abeja) | Processing Times (2.2/3) |
| ------------------------- | ----------------- | ---------- | --------------- | ---------------- | ----------------- | ------------------------ |
| 20,001                    | -                | 81MB      | 5,589,514              | -      | -       | -       |

### Data Paths

- Pathes under `/bb/grandchallenge/gaf51090/datasets/redpajama_github`

| Language | Raw Data                     | Processed jsonl files (after step 2) | Binarized Data (Ours Ver.1)         | Binarized Data (GPT-2)             | Binarized Data (Abeja) |
| -------- | --------------------------- | ----------------------------------- | ----------------------------- | --------------------------------- | --------------------- |
| Japanese | -   | merged/merged.jsonl        | binarized/sentencepiece_ver1/redpajama_github | -       |  - |





