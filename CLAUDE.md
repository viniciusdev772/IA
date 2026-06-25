# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational PT-BR language model built from scratch. Custom byte-level BPE tokenizer + Gemma-3-inspired transformer (GemmaMicro). Python 3.12+, PyTorch, GPU (CUDA).

## Architecture

GemmaMicro mirrors Gemma 3 at micro scale:
- 12 layers, d_model=256, 8 heads, ctx_len=512
- Alternating local(5):global(1) attention pattern per block
- Local: sliding window (64 tokens), RoPE base=10K
- Global: full causal attention, RoPE base=1M
- SwiGLU FFN (8/3 × d_model, rounded to multiple of 64)
- RMSNorm (not LayerNorm)
- Weight tying: tok_emb ↔ lm_head

## Tokenizer

BPE byte-level (vocab base = 256 bytes + merges). Special tokens: `<pad>=256, <bos>=257, <eos>=258, <user>=259, <asst>=260`. Training uses HuggingFace `tokenizers` Rust backend for speed, but the wrapper class is custom Python.

## Dialog Training

Format: `<bos> <user> pergunta <asst> resposta <eos>`. Loss masking: loss computed only on assistant tokens (after `<asst>`). Dialog files use pipe `|` separator: `usuário: oi|assistente: olá!`.

## Running

```bash
# train (runs both LM + dialog training, saves to gemma_micro/)
python nn.py

# load trained model in Python
from nn import load_model
model, tokenizer = load_model("gemma_micro")
```

## Dependencies

`torch`, `tokenizers` (HuggingFace), `requests`, `trafilatura`. No requirements.txt — install manually.

## File Structure

- `nn.py` — tokenizer, model, training loop, generation, CLI entry point
- `crawl.py` — web crawler for PT-BR text (Wikipedia, G1, UOL, Agência Brasil)
- `clean_crawl.py` — post-processing/cleaning of crawled data
- `dataset_*.txt`, `frases_treinamento.txt`, `dialogs_pt.txt` — training data
- `gemma_micro/` — saved model checkpoint + tokenizer
- `gpt2_mini/` — older/smaller model checkpoint

## Code Conventions

- All code, comments, and docstrings in Portuguese (Brazilian)
- Single-file architecture: `nn.py` contains everything (tokenizer, model, datasets, training)
- No test suite, no linter, no formatter configured
