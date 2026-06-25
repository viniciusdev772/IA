"""Teste rápido de treino CPU vs GPU com monitoramento de memória."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import psutil
import os
import time
import math

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from nn import (
    BPETokenizer, GemmaMicro, NextWordDataset, DialogDataset,
    CollateFn, load_texts,
)


def mem_stats(device: str) -> str:
    ram = psutil.Process().memory_info().rss / 1024**2
    if device == "cuda":
        alloc = torch.cuda.memory_allocated() / 1024**2
        peak = torch.cuda.max_memory_allocated() / 1024**2
        return f"RAM={ram:.0f}MB | VRAM alloc={alloc:.0f}MB peak={peak:.0f}MB"
    return f"RAM={ram:.0f}MB"


def test_device(device: str, epochs: int = 2, batch_size: int = 8, accum_steps: int = 4):
    print(f"\n{'='*60}")
    print(f"  TESTE: {device.upper()}")
    print(f"{'='*60}")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    use_amp = device == "cuda"
    print(f"AMP: {use_amp} | batch={batch_size} | accum={accum_steps} | effective={batch_size*accum_steps}")
    print(f"Antes do treino: {mem_stats(device)}")

    # tokenizer
    t0 = time.time()
    texts = load_texts("dataset_portugues_br.txt", "frases_treinamento.txt")
    if not texts:
        print("ERRO: sem dados de treino")
        return
    tokenizer = BPETokenizer(vocab_size=8000).train(texts)
    print(f"Tokenizer treinado em {time.time()-t0:.1f}s | vocab={tokenizer.vocab_size}")
    print(f"Após tokenizer: {mem_stats(device)}")

    # dataset
    ctx_len = 128
    dataset = NextWordDataset(texts, tokenizer, ctx_len=ctx_len)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, persistent_workers=True,
        pin_memory=(device == "cuda"),
        collate_fn=CollateFn(tokenizer.pad_id),
    )
    print(f"Dataset: {len(dataset)} amostras, {len(loader)} batches")

    # modelo
    model = GemmaMicro(vocab_size=tokenizer.vocab_size, ctx_len=ctx_len).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Modelo: {n_params:,} params")
    print(f"Após modelo.to({device}): {mem_stats(device)}")

    # torch.compile (só GPU — CPU é lento demais na compilação)
    if device == "cuda":
        print("Compilando modelo (torch.compile)...")
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # treino
    print(f"\nIniciando {epochs} epochs...")
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        t_epoch = time.time()

        for batch_idx, (X, Y) in enumerate(loader):
            X, Y = X.to(device), Y.to(device)
            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(X)
                loss = F.cross_entropy(
                    logits.view(-1, tokenizer.vocab_size),
                    Y.view(-1),
                    ignore_index=tokenizer.pad_id,
                )
                loss = loss / accum_steps

            scaler.scale(loss).backward()
            total_loss += loss.item() * accum_steps

            if (batch_idx + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        avg_loss = total_loss / len(loader)
        ppl = math.exp(min(avg_loss, 20))
        elapsed = time.time() - t_epoch
        samples_sec = len(dataset) / elapsed
        print(f"  Epoch {epoch}/{epochs} | loss={avg_loss:.4f} | ppl={ppl:.1f} | "
              f"{elapsed:.1f}s ({samples_sec:.0f} samples/s) | {mem_stats(device)}")

    total_time = time.time() - t_start
    print(f"\n  RESULTADO {device.upper()}:")
    print(f"    Tempo total: {total_time:.1f}s")
    print(f"    Loss final: {avg_loss:.4f}")
    print(f"    Perplexidade: {ppl:.1f}")
    print(f"    Throughput: {len(dataset)*epochs/total_time:.0f} samples/s médio")
    print(f"    Memória final: {mem_stats(device)}")

    # geração rápida
    model.eval()
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    prompt = "a inteligência artificial"
    ids = tokenizer.encode(prompt, add_special=False)
    out = raw.generate(ids, max_new=15, temperature=0.8)
    print(f"    Geração: [{prompt}] → {tokenizer.decode(out)}")

    # cleanup
    del model, optimizer, scaler, loader
    if device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    print("="*60)
    print("  TESTE DE TREINO — CPU vs GPU")
    print(f"  PyTorch {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  RAM: {psutil.virtual_memory().total/1024**3:.1f}GB")
    print("="*60)

    # CPU test (batch menor, sem compile)
    test_device("cpu", epochs=2, batch_size=8, accum_steps=2)

    # GPU test
    if torch.cuda.is_available():
        test_device("cuda", epochs=2, batch_size=16, accum_steps=8)

    print("\n" + "="*60)
    print("  TESTES CONCLUÍDOS")
    print("="*60)
