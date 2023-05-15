# Ameba Blog Data Preparation

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

### 3. Tokenize and binarize data into the megatron-deepspeed format
**Sentencepiece (ours):**
```bash
export OUTDIR=/bb/grandchallenge/gaf51090/datasets/ameba_blog/binarized/sentencepiece_ver1
mkdir -p $OUTDIR
```

Tokenize and binarize:
```bash
python tools/preprocess_data.py \
        --input /bb/grandchallenge/gaf51090/datasets/ameba_blog/processed/entry_text.jsonl \
        --output-prefix $OUTDIR/abema_blog \
        --vocab-file /bb/grandchallenge/gaf51090/datasets/tokenization_replaced/spm_input_fall_replaced_all.model \
        --dataset-impl mmap \
        --tokenizer-type JapaneseSentencePiece \
        --workers 64 \
        --append-eod
```

## Dataset Statistics

Here's the formatted table:

### Basic Statistics

- Processing time calculated using `rt_C.small=1`
- (†) uses `login-node`

| # Extracted Blogs | # Discarded Blogs | Jsonl Size | # Tokens (Ours Ver.1) | # Tokens (GPT-2) | # Tokens (Abeja) | Processing Times (2.2/3) |
| ------------------------- | ----------------- | ---------- | --------------- | ---------------- | ----------------- | ------------------------ |
| 20,000                    | -                | 81MB      | 5,589,514              | -      | -       | -       |

### Data Paths

- Pathes under `/bb/grandchallenge/gaf51090/datasets/ameba_blog`

| Language | Raw Data                     | Processed jsonl files (after step 2) | Binarized Data (Ours Ver.1)         | Binarized Data (GPT-2)             | Binarized Data (Abeja) |
| -------- | --------------------------- | ----------------------------------- | ----------------------------- | --------------------------------- | --------------------- |
| Japanese | -   | processed/entry_text.jsonl        | binarized/sentencepiece_ver1/abema_blog | -       |  - |





