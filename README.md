# BVV241 Tokenizer Benchmarking & Frozen Embedding Sets

A research resource for cross-model Unicode-centric tokenization, *frozen embedding* LMs, and experimentation on the emergence of semantics and modular model fusion.

## About This Repository

This repo provides:

- Scripts, notebooks, and raw files for construction and benchmarking of *Unicode-based tokenizers* with n-gram/Wikipedia statistical enrichment,
- Precomputed, **L2-normalized, frozen embedding matrices** (for direct plug-in as nn.Embedding),
- Tools for building hybrid vocabularies (Unicode + bigram/trigram extensions + SOTA token string intersection),
- Live benchmarking and visualization pipelines (SOTA vs custom models, t-SNE, BLEU/MMLU/ARC),
- **HuggingFace Hub** integration, with all resources ([Bochkov](https://huggingface.co/Bochkov)).

This work enables research on:

- *Semantic emergence* in LMs trained on non-semantic, fixed surface embeddings,
- Unified and modular architectures for multi-expert, multilingual, or scalable models,
- Experimental fusion of models with different tokenizer histories.


## Tokenizer and Embedding Variants

### 1. [bvv241-2-3](https://huggingface.co/Bochkov/bvv241-2-3)
- **Unicode plane** (0–65535): All single Unicode codepoints (monograms).
- **Private/unused Unicode ranges**: Wikipedia bigrams/trigrams.
- **Vocabulary**: 65,536 tokens; **Embedding**: 1024-dim, L2-normalized, frozen (**no semantics**).
- Suitable for: *Baseline Unicode LM research, non-semantic embedding experiments.*

### 2. [bvv241-max](https://huggingface.co/Bochkov/bvv241-max)
- **Unicode monograms** + bigrams/trigrams + *intersection of token strings* across SOTA models (o200k_base, cl100k_base, Mistral-Nemo, DeepSeek-R1, etc).
- **Vocabulary**: 131,072 tokens; **Embedding**: 1024-dim, frozen.
- Suitable for: *Unified tokenizer/embedding research; plug-and-play fusion across SOTA models.*

### 3. [bvv241-nemo](https://huggingface.co/Bochkov/bvv241-nemo)
- Vocabulary of Mistral-Nemo SOTA model with frozen *surface-level* (non-semantic) embeddings.
- **Vocabulary**: 131,072; **Embedding**: 1024-dim, frozen.
- Suitable for: *Direct Mistral-Nemo token/embedding comparison.*

### 4. [bvv241-abs](https://huggingface.co/Bochkov/bvv241-abs)
- As `bvv241-max`, but **embedding size 4096**.
- Suitable for: *Experiments on scaling embedding space.*

**All embedding matrices:** L2-normalized, *fixed/frozen*, contain **no semantic information**.

---

## Quick Start

from transformers import AutoTokenizer

from huggingface_hub import hf_hub_download

import torch

tokenizer = AutoTokenizer.from_pretrained('Bochkov/bvv241-max')

emb_path = hf_hub_download(repo_id="Bochkov/bvv241-max", filename="normalized_embeddings_weights.pt")

embeddings = torch.load(emb_path)  # shape: [vocab_size, emb_dim]

## 📊 Benchmarks & Research Notebooks
**_tokenizer-benchmarking-t-sne.ipynb**

— Visualizes token/embedding distribution via t-SNE, comparing BVV tokenizers with SOTA baselines.


**_models_benchmarking.py, _models_benchmarking.plot.ipynb, _models_benchmarking.code.ipynb**

— Scripts and notebooks to benchmark models (BLEU, MMLU, ARC) using these tokenizers & embeddings versus SOTA tokenizers.


**_n-gramms-from-wiki.ipynb**

— Extraction of frequent n-grams from Wikipedia to fill Unicode private ranges, enriching token coverage.


**_tokenizer-builder-*.ipynb**

— Complete construction logic for each tokenizer/embedding variant.

## 🗂️ File Structure
### File/Notebook	Purpose
_tokenizer-benchmarking-t-sne.ipynb	- t-SNE visualizations of token space and embedding overlap

_n-gramms-from-wiki.ipynb	- Extracting n-grams for vocab extension

_n-gramms-2-3-4-5.txt (etc)	- Precomputed n-gram lists (for reproducible vocab)

_n-gramms-intersection.txt	- Common token strings across SOTA tokenizers

_tokenizer-builder-*	- Jupyter code for building each tokenizer/embedding set

_models_benchmarking.*	- Benchmark scripts, plots, example use in LM evaluation

normalized_embeddings_weights.pt	- Main embedding matrix for each tokenizer version

All tokenizers and embeddings are mirrored on HuggingFace Hub.

## ⚗️ Research Scope & Scientific Context
## Purpose:

These resources enable:

Investigation into semantic emergence when training transformers with fixed, non-semantic ("surface-level") embeddings.

Plug-and-play modular/MoE experiments: plug-in new "experts" or fuse LMs trained with different tokenizations, since embeddings are structurally identical and fixed.

Exploration of Unicode-standardized, reproducible vocabularies for multilingual and cross-model pipelines.

Scientific novelty:

These embeddings are never trained, encode no semantic information, and are suitable for research into meaning arising solely in transformer layers above embedding.

You are free to combine, swap, and experiment with models and tokenizers—unified by their Unicode-surface-based, frozen embedding matrix—without retraining! This enables fair benchmarking of the ability of deep architectures (alone) to synthesize meaning.

## 🧑‍🔬 Citation & Concept
🧑‍🔬 Citation & Concept

If you use or build upon this demo, please cite:
```
@misc{bochkov2025emergentsemanticstokenembeddings,
      title={Emergent Semantics Beyond Token Embeddings: Transformer LMs with Frozen Visual Unicode Representations}, 
      author={A. Bochkov},
      year={2025},
      eprint={2507.04886},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.04886}, 
}

```

Feel free to contact for research collaborations.


