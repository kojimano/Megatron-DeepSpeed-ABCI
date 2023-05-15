# Aozora Books Data Preparation

## Table of Contents
- [Procedures](#procedures)
  * [1. Get computes (ABCI) and install packages](#1-get-computes-abci-and-install-packages)
  * [2. Download, clean and format the latest Aozara book data](#2-download-clean-and-format-the-latest-aozara-book-data)
  * [3. Tokenize and binarize data into the megatron-deepspeed format](#3-tokenize-and-binarize-data-into-the-megatron-deepspeed-format)
- [Dataset Statistics](#dataset-statistics)
  * [Basic Statistics](#basic-statistics)
  * [Data Paths](#data-paths)
- [References](#references)

## Procedures

### 1. Get computes (ABCI) and install packages
```bash
# 1.1 Get computes
qrsh -g gaf51090 -l rt_C.small=1 -l h_rt=2:00:00 

# 1.2 Activate your virtual environment
source XXX/bin/activate

# 1.3 Install wikiextractor
pip install bs4 lxml tqdm
```

### 2. Download, clean and format the latest Aozara book data
```bash
# 2.1 Download data
git clone --branch master --depth 1 https://github.com/aozorabunko/aozorabunko.git /bb/grandchallenge/gaf51090/datasets/aozora_books/raw_data/aozorabunko

# 2.2 Generate data files with a loose JSON (jsonl) format
python -m abci.dataset.aozora_books.extract_jsonl
```

*Filtering and cleaning details: [TBD]*

### 3. Tokenize and binarize data into the megatron-deepspeed format

**Sentencepiece (nmt_nfkc):**
```bash
export OUTDIR=/bb/grandchallenge/gaf51090/datasets/aozora_books/binarized/sentencepiece_ver1
mkdir -p $OUTDIR

python tools/preprocess_data.py \
        --input /bb/grandchallenge/gaf51090/datasets/aozora_books/processed/aozora_books.jsonl \
        --output-prefix $OUTDIR/aozora_books \
        --vocab-file /bb/grandchallenge/gaf51090/datasets/tokenization_replaced/spm_input_fall_replaced_all.model \
        --dataset-impl mmap \
        --tokenizer-type JapaneseSentencePiece \
        --workers 64 \
        --append-eod
```

**Sentencepiece (nmt_nfkc) \w ws & tab:**
```bash
export OUTDIR=/bb/grandchallenge/gaf51090/datasets/aozora_books/binarized/sp_nmt_nfkc_with_ws_tab
mkdir -p $OUTDIR
export MODELDIR=/bb/grandchallenge/gaf51090/tokenizer_new/spm_input_fall_replaced_all_wodummyprefix_modified.model

python tools/preprocess_data.py \
        --input /bb/grandchallenge/gaf51090/datasets/aozora_books/processed/aozora_books.jsonl \
        --output-prefix $OUTDIR/aozora_books \
        --vocab-file $MODELDIR\
        --dataset-impl mmap \
        --tokenizer-type JapaneseSentencePiece \
        --workers 64 \
        --append-eod
```

**Sentencepiece (identity):**
```bash
export OUTDIR=/bb/grandchallenge/gaf51090/datasets/aozora_books/binarized/sp_identity
mkdir -p $OUTDIR
export MODELDIR=/bb/grandchallenge/gaf51090/tokenizer_new/spm_input_fall_replaced_all_identity_wodummyprefix_modified.model

python tools/preprocess_data.py \
        --input /bb/grandchallenge/gaf51090/datasets/aozora_books/processed/aozora_books.jsonl \
        --output-prefix $OUTDIR/aozora_books \
        --vocab-file $MODELDIR \
        --dataset-impl mmap \
        --tokenizer-type JapaneseSentencePiece \
        --workers 64 \
        --append-eod
```

**OpenAI GPT-2:**
```bash
export OUTDIR=/bb/grandchallenge/gaf51090/datasets/aozora_books/binarized/gpt-2
mkdir -p $OUTDIR

python tools/preprocess_data.py \
        --input /bb/grandchallenge/gaf51090/datasets/aozora_books/processed/aozora_books.jsonl \
        --output-prefix $OUTDIR/aozora_books \
        --vocab-file dataset/gpt2-vocab.json \
        --merge-file dataset/gpt2-merges.txt \
        --dataset-impl mmap \
        --tokenizer-type GPT2BPETokenizer \
        --workers 64 \
        --append-eod
```

**Abeja Japanese GPTNeoX:**
```bash
export OUTDIR=/bb/grandchallenge/gaf51090/datasets/aozora_books/binarized/abeja
mkdir -p $OUTDIR
pip install transformers
pip install sentencepiece

python tools/preprocess_data.py \
        --input /bb/grandchallenge/gaf51090/datasets/aozora_books/processed/aozora_books.jsonl \
        --output-prefix $OUTDIR/aozora_books \
        --dataset-impl mmap \
        --tokenizer-type AbejaJapaneseGPT2Tokenizer \
        --workers 1 \
        --append-eod
```

## Dataset Statistics

Here's the formatted table:


### Dataset
| # Extracted HTMLs (Books) | Jsonl Size | Processing Times (2.2/3) |
| -------- | -------------------- | ----------------- |--------------- | 
| 2,219,610            | 6.9 GB            |  38 mins / 1 <mins / 70<? mins |

- Processing time calculated using `rt_C.small=1`
- (†) uses `rt_C.large=1`

### Number of Tokens
| SP (nmt_nfkc) | SP (nmt_nfkc) \w ws & tab | SP (identity) |GPT-2 | Abeja | 
| -------- | -------------------- | ---------------------- |---------------- |---------------- | ---------------- | 
| 153,122,906          | 153,183,321      | 153,672,388     |351,867,040    |177,835,717      |   


### Data Paths
- Pathes under `/bb/grandchallenge/gaf51090/datasets/aozora_books`
**Raw / Processd / Merged Data**
| Raw Data       | Processed jsonl files (after step 2) | 
| ------------------------- | ------------------------------------- | --------------------------- | 
| raw_data/aozorabunko/cards | rocessed/aozora_books.jsonl           | 

**Binary**
| SP (nmt_nfkc) | SP (nmt_nfkc) \w ws & tab | SP (identity) |GPT-2 | Abeja | 
| -------- | -------------------- | ---------------------- |---------------- |---------------- | ---------------- | 
| binarized/sentencepiece_ver1  | binarized/sp_nmt_nfkc_with_ws_tab | binarized/sp_identity  | binarized/gpt-2 | binarized/abeja |   

## References
- [Scraping](https://qiita.com/Yupine/items/92d75865a72c60ae7285)
- [Preprocessing](https://qiita.com/y_itoh/items/fa04c1e2f3df2e807d61)

