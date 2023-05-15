# Wikipedia Data Preparation

This README describes the process of downloading, cleaning, formatting, tokenizing, and binarizing Wikipedia data in Japanese and English.

## Table of Contents
- [Procedures](#procedures)
  * [1. Get computes (ABCI) and install packages](#1-get-computes-abci-and-install-packages)
  * [2. Download, clean and format the latest Wikipedia (ja, en) dumps](#2-download-clean-and-format-the-latest-wikipedia-ja-en-dumps)
  * [3. Tokenize and binarize data into the megatron-deepspeed format](#3-tokenize-and-binarize-data-into-the-megatron-deepspeed-format)
- [Dataset Statistics](#dataset-statistics)
  * [Basic Statistics](#basic-statistics)
  * [Data Paths](#data-paths)

## Procedures

### 1. Get computes (ABCI) and install packages

```bash
# 1.1 get computes
qrsh -g gaf51090 -l rt_C.small=1 -l h_rt=2:00:00 

# 1.2 activate your virtual environment
source XXX/bin/activate

# 1.3 install wikiextractor
pip install wikiextractor
```

### 2. Download, clean and format the latest Wikipedia (ja, en) dumps

```bash
# generate data files with a loose json (jsonl) format (each file is 100MB)
python -m abci.dataset.wikipedia.wikidump_download
```

## 3. Tokenize and binarize data into the megatron-deepspeed format

To tokenize and binarize the data, follow these steps:

### 3.1. Merge smaller jsonl files into a single jsonl:

   **Japanese:**

   ```
   ./abci/dataset/wikipedia/merge_files.sh /bb/grandchallenge/gaf51090/datasets/wikipedia/processed/ja/AA /bb/grandchallenge/gaf51090/datasets/wikipedia/merged/ja ja_merged
   ```

   **English:**

   ```
   ./abci/dataset/wikipedia/merge_files.sh /bb/grandchallenge/gaf51090/datasets/wikipedia/processed/en/AA /bb/grandchallenge/gaf51090/datasets/wikipedia/merged/en en_merged_1
   ./abci/dataset/wikipedia/merge_files.sh /bb/grandchallenge/gaf51090/datasets/wikipedia/processed/en/AB /bb/grandchallenge/gaf51090/datasets/wikipedia/merged/en en_merged_2
   ./abci/dataset/wikipedia/merge_files.sh /bb/grandchallenge/gaf51090/datasets/wikipedia/merged/en /bb/grandchallenge/gaf51090/datasets/wikipedia/merged/en en_merged
   ```

### 3.2. Tokenize and binarize data:
   **Sentencepiece (nmt_nfkc):**
   ```bash
   # Set the output directory:
   export OUTDIR=/bb/grandchallenge/gaf51090/datasets/wikipedia/binarized/sentencepiece_ver1
   mkdir -p $OUTDIR
   export MODELDIR=/bb/grandchallenge/gaf51090/datasets/tokenization_replaced/spm_input_fall_replaced_all.model

   # Tokenize and binarize Japanese
   python tools/preprocess_data.py \
          --input /bb/grandchallenge/gaf51090/datasets/wikipedia/merged/ja/ja_merged.json \
          --output-prefix $OUTDIR/ja_wiki \
          --vocab-file $MODELDIR \
          --dataset-impl mmap \
          --tokenizer-type JapaneseSentencePiece \
          --workers 64 \
          --append-eod

   # Tokenize and binarize English
   python tools/preprocess_data.py \
          --input /bb/grandchallenge/gaf51090/datasets/wikipedia/merged/en/en_merged.json \
          --output-prefix $OUTDIR/en_wiki \
          --vocab-file $MODELDIR \
          --dataset-impl mmap \
          --tokenizer-type JapaneseSentencePiece \
          --workers 64 \
          --append-eod
   ```


   **Sentencepiece (nmt_nfkc) \w ws & tab :**

   Set the output directory:

   ```bash
   # Set the output directory
   export OUTDIR=/bb/grandchallenge/gaf51090/datasets/wikipedia/binarized/sp_nmt_nfkc_with_ws_tab
   mkdir -p $OUTDIR
   export MODELDIR=/bb/grandchallenge/gaf51090/tokenizer_new/spm_input_fall_replaced_all_wodummyprefix_modified.model

   # Tokenize and binarize Japanese
   python tools/preprocess_data.py \
          --input /bb/grandchallenge/gaf51090/datasets/wikipedia/merged/ja/ja_merged.json \
          --output-prefix $OUTDIR/ja_wiki \
          --vocab-file $MODELDIR \
          --dataset-impl mmap \
          --tokenizer-type JapaneseSentencePiece \
          --workers 64 \
          --append-eod
   #  Tokenize and binarize English
   python tools/preprocess_data.py \
          --input /bb/grandchallenge/gaf51090/datasets/wikipedia/merged/en/en_merged.json \
          --output-prefix $OUTDIR/en_wiki \
          --vocab-file  $MODELDIR \
          --dataset-impl mmap \
          --tokenizer-type JapaneseSentencePiece \
          --workers 64 \
          --append-eod

   ```

   **Sentencepiece (identity):**
   ```bash
   # Set the output directory
   export OUTDIR=/bb/grandchallenge/gaf51090/datasets/wikipedia/binarized/sp_identity
   mkdir -p $OUTDIR
   export MODELDIR=/bb/grandchallenge/gaf51090/tokenizer_new/spm_input_fall_replaced_all_identity_wodummyprefix_modified.model

   # Tokenize and binarize Japanese
   python tools/preprocess_data.py \
       --input /bb/grandchallenge/gaf51090/datasets/wikipedia/merged/ja/ja_merged.json \
       --output-prefix $OUTDIR/ja_wiki \
       --vocab-file $MODELDIR \
       --dataset-impl mmap \
       --tokenizer-type JapaneseSentencePiece \
       --workers 64 \
       --append-eod

   # Tokenize and binarize English
   python tools/preprocess_data.py \
          --input /bb/grandchallenge/gaf51090/datasets/wikipedia/merged/en/en_merged.json \
          --output-prefix $OUTDIR/en_wiki \
          --vocab-file  $MODELDIR \
          --dataset-impl mmap \
          --tokenizer-type JapaneseSentencePiece \
          --workers 64 \
          --append-eod
   ```

   **GPT-2:**

   Set the output directory:

   ```bash
   # Set the output directory
   export OUTDIR=/bb/grandchallenge/gaf51090/datasets/wikipedia/binarized/gpt-2
   mkdir -p $OUTDIR

   # Tokenize and binarize Japanese
   python tools/preprocess_data.py \
          --input /bb/grandchallenge/gaf51090/datasets/wikipedia/merged/ja/ja_merged.json \
          --output-prefix $OUTDIR/ja_wiki \
          --vocab-file dataset/gpt2-vocab.json \
          --merge-file dataset/gpt2-merges.txt \
          --dataset-impl mmap \
          --tokenizer-type GPT2BPETokenizer \
          --workers 16 \
          --append-eod

   # Tokenize and binarize English
   python tools/preprocess_data.py \
          --input /bb/grandchallenge/gaf51090/datasets/wikipedia/merged/en/en_merged.json \
          --output-prefix $OUTDIR/en_wiki \
          --vocab-file dataset/gpt2-vocab.json \
          --merge-file dataset/gpt2-merges.txt \
          --dataset-impl mmap \
          --tokenizer-type GPT2BPETokenizer \
          --workers 64 \
          --append-eod
   ```

   **Abeja Japanese GPTNeoX:**

   ```bash
   # Set the output directory
   export OUTDIR=/bb/grandchallenge/gaf51090/datasets/wikipedia/binarized/abeja
   mkdir -p $OUTDIR

   # Tokenize and binarize Japanese
   python tools/preprocess_data.py \
          --input /bb/grandchallenge/gaf51090/datasets/wikipedia/merged/ja/ja_merged.json \
          --output-prefix $OUTDIR/ja_wiki \
          --dataset-impl mmap \
          --tokenizer-type AbejaJapaneseGPT2Tokenizer \
          --workers 16 \
          --append-eod

   # Tokenize and binarize English
   python tools/preprocess_data.py \
          --input /bb/grandchallenge/gaf51090/datasets/wikipedia/merged/en/en_merged.json \
          --output-prefix $OUTDIR/en_wiki \
          --dataset-impl mmap \
          --tokenizer-type AbejaJapaneseGPT2Tokenizer \
          --workers 64 \
          --append-eod
   ```

## Dataset Statistics


### Dataset
| Language | # Extracted Articles | Merged Jsonl Size | Processing Times (2/3.1/3.2) |
| -------- | -------------------- | ----------------- |--------------- | 
| Japanese | 2,219,610            | 6.9 GB            |  38 mins / 1 <mins / 70<? mins |
| English  | 17,020,965           | 17.4 GB           |  208 mins / 1 <mins / 15† mins   |

- Processing time calculated using `rt_C.small=1`
- (†) uses `rt_C.large=1`

### Number of Tokens
| Language | SP (nmt_nfkc) | SP (nmt_nfkc) \w ws & tab | SP (identity) |GPT-2 | Abeja | 
| -------- | -------------------- | ---------------------- |---------------- |---------------- | ---------------- | 
| Japanese | 673,848,504          | 672,770,579            | 676,288,688     |1,802,750,651    | 948,134,289      |   
| English  | 4,107,945,372        | 4,113,934,495          | 4,133,476,592  |3,517,216,353 | 15,686,907,144   |   


### Data Paths

- Pathes under `/bb/grandchallenge/gaf51090/datasets/wikipedia`
**Raw / Processd / Merged Data**
| Language | Compressed Raw Data       | Processed jsonl files (after step 2) | Merged jsonl (after step 3.1)|
| -------- | ------------------------- | ------------------------------------- | --------------------------- | 
| Japanese | raw_data/ja/ja_xml.bz2 | processed/ja/AA            | merged/ja/ja_merged.json  |
| English  | raw_data/en/en_xml.bz2 | processed/en/AA, processed/en/AB         | merged/en/en_merged.json  | 

**Binary**
| SP (nmt_nfkc) | SP (nmt_nfkc) \w ws & tab | SP (identity) |GPT-2 | Abeja | 
| -------- | -------------------- | ---------------------- |---------------- |---------------- | ---------------- | 
| binarized/sentencepiece_ver1  | binarized/sp_nmt_nfkc_with_ws_tab | binarized/sp_identity  | binarized/gpt-2 | binarized/abeja |   

