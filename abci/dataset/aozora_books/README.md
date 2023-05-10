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
**OpenAI GPT-2:**
```bash
export OUTDIR=/bb/grandchallenge/gaf51090/datasets/aozora_books/binarized/gpt-2
mkdir -p $OUTDIR
```

Tokenize and binarize:
```bash
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
**Japanese GPTNeoX:**
```bash
export OUTDIR=/bb/grandchallenge/gaf51090/datasets/aozora_books/binarized/japanese_gptneox
mkdir -p $OUTDIR

pip install transformers
pip install sentencepiece
```

Tokenize and binarize:
```bash
python tools/preprocess_data.py \
        --input /bb/grandchallenge/gaf51090/datasets/aozora_books/processed/aozora_books.jsonl \
        --output-prefix $OUTDIR/aozora_books \
        --vocab-file /bb/grandchallenge/gaf51090/tokenizer/abeja_japanese_sp.model \
        --dataset-impl mmap \
        --tokenizer-type JapaneseGPT2Tokenizer \
        --workers 64 \
        --append-eod
```


## Dataset Statistics

### Basic Statistics

- Processing time calculated using `rt_C.small=1`
- (†) uses `login-node`

| # Extracted HTMLs (Books) | # Discarded HTMLs | Jsonl Size | # Tokens (GPT-2) | # Tokens (Abeja) | Processing Times (2.2/3) |
|---------------------------|-------------------|------------|------------------|------------------|--------------------------|
| 17,383                    | 47                | 1.3GB      | 351,867,040      |  155,676,115               |15† mins / 3† mins        |


### Data Paths

- Pathes under `/bb/grandchallenge/gaf51090/datasets/aozora_books`

| Language | Raw Data       | Processed jsonl files (after step 2) | Binarized Data (GPT-2) | Binarized Data (Rinna) |
| -------- | ------------------------- | ------------------------------------- |  --------------------- | --------------------- |
| Japanese | raw_data/aozorabunko/cards | processed/aozora_books.jsonl   | binarized/gpt-2/aozora_books | -                     |


## References
- [Scraping](https://qiita.com/Yupine/items/92d75865a72c60ae7285)
- [Preprocessing](https://qiita.com/y_itoh/items/fa04c1e2f3df2e807d61)


