"""
GPT-2 com BPE byte-level tokenizer próprio e RoPE positional encoding.
Treina previsão de próxima palavra em português brasileiro.

Tokenizer: BPE byte-level (estilo minbpe/karpathy)
- vocab base: 256 bytes
- merges iterativos até vocab_size alvo
- zero UNK: qualquer UTF-8 é representável como bytes
- tokens especiais: <pad>=256, <bos>=257, <eos>=258
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import re
import tempfile
from pathlib import Path
from collections import Counter
import math

# pre-tokenização: separa palavras, pontuação e espaço+palavra (estilo GPT-2)
_PRETOK = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+|[^\s\w]+|\s+(?!\S)|\s", re.UNICODE)


# ─── BPE Tokenizer byte-level ──────────────────────────────────────────────

def _get_pairs(ids: list[int]) -> Counter:
    """Conta todos os pares adjacentes."""
    c: Counter = Counter()
    for a, b in zip(ids, ids[1:]):
        c[(a, b)] += 1
    return c


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Substitui todas as ocorrências de pair por new_id."""
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    PAD_ID  = 256
    BOS_ID  = 257
    EOS_ID  = 258
    BASE_VOCAB  = 256          # bytes 0-255
    NUM_SPECIAL = 3            # PAD, BOS, EOS

    def __init__(self, vocab_size: int = 8000):
        assert vocab_size > self.BASE_VOCAB + self.NUM_SPECIAL
        self.vocab_size = vocab_size
        # merges: lista ordenada de (pair -> new_id)
        self.merges: dict[tuple[int, int], int] = {}
        # id -> bytes
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(self.BASE_VOCAB)}
        self.vocab[self.PAD_ID]  = b"<pad>"
        self.vocab[self.BOS_ID]  = b"<bos>"
        self.vocab[self.EOS_ID]  = b"<eos>"

    def _chunks(self, text: str) -> list[bytes]:
        """Pre-tokeniza por palavra/pontuação; BPE nunca cruza fronteira."""
        return [m.group().encode("utf-8") for m in _PRETOK.finditer(text.lower())]

    def train(self, texts: list[str]) -> "BPETokenizer":
        """Treina BPE via tokenizers HF (Rust) — ordens de magnitude mais rápido."""
        from tokenizers import Tokenizer as HFTokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder

        sample = texts
        n_merges = self.vocab_size - self.BASE_VOCAB - self.NUM_SPECIAL
        print(f"  BPE (Rust): {len(sample)} textos, vocab_size={self.vocab_size}, merges={n_merges}", flush=True)

        hf_tok = HFTokenizer(BPE(unk_token=None))
        hf_tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
        hf_tok.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=[],
            min_frequency=2,
            show_progress=False,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            for t in sample:
                f.write(t.lower())
                f.write("\n")
            tmp_path = f.name

        hf_tok.train([tmp_path], trainer)
        Path(tmp_path).unlink()

        # extrai merges e vocab via JSON serialization
        import json as _json
        data = _json.loads(hf_tok.to_str())
        hf_vocab  = data["model"]["vocab"]   # str -> hf_id
        hf_merges = data["model"]["merges"]  # [[a_str, b_str], ...]

        # GPT-2 byte decoder: chr -> byte value
        bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        _byte_decoder: dict[str, int] = {chr(c): b for b, c in zip(bs, cs)}

        def hf_str_to_bytes(s: str) -> bytes:
            return bytes([_byte_decoder.get(c, ord(c) % 256) for c in s])

        # str->bytes para todos os tokens HF
        hf_id_to_bytes: dict[int, bytes] = {
            hf_id: hf_str_to_bytes(tok_str)
            for tok_str, hf_id in hf_vocab.items()
        }

        # bytes->nosso_id (vocab base: byte i -> id i)
        bytes_to_our_id: dict[bytes, int] = {
            bytes([i]): i for i in range(self.BASE_VOCAB)
        }

        next_id = self.BASE_VOCAB + self.NUM_SPECIAL
        for merge_pair in hf_merges:
            if next_id >= self.vocab_size:
                break
            a_str, b_str = merge_pair
            a_bytes = hf_str_to_bytes(a_str)
            b_bytes = hf_str_to_bytes(b_str)
            merged  = a_bytes + b_bytes

            a_id = bytes_to_our_id.get(a_bytes)
            b_id = bytes_to_our_id.get(b_bytes)
            if a_id is None or b_id is None:
                continue

            self.merges[(a_id, b_id)] = next_id
            self.vocab[next_id] = merged
            bytes_to_our_id[merged] = next_id
            next_id += 1

        print(f"  BPE: vocab final {next_id} ({len(self.merges)} merges)")

        # guarda HF tokenizer + mapa hf_id -> nosso_id para encode rápido
        self._hf_tok = hf_tok
        hf_to_our: dict[int, int] = {}
        for tok_str, hf_id in hf_vocab.items():
            b = hf_str_to_bytes(tok_str)
            our_id = bytes_to_our_id.get(b)
            if our_id is not None:
                hf_to_our[hf_id] = our_id
        self._hf_to_our = hf_to_our

        return self

    def _encode_chunk(self, b: bytes) -> list[int]:
        ids = list(b)
        for pair, new_id in self.merges.items():
            ids = _merge(ids, pair, new_id)
        return ids

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        # caminho rápido: usa HF tokenizer Rust se disponível
        if hasattr(self, "_hf_tok") and self._hf_tok is not None:
            enc = self._hf_tok.encode(text.lower())
            ids = [self._hf_to_our.get(hf_id, 0) for hf_id in enc.ids]
        else:
            ids = []
            for chunk in self._chunks(text):
                ids.extend(self._encode_chunk(chunk))
        if add_special:
            ids = [self.BOS_ID] + ids + [self.EOS_ID]
        return ids

    def encode_batch(self, texts: list[str], add_special: bool = True) -> list[list[int]]:
        if hasattr(self, "_hf_tok") and self._hf_tok is not None:
            lowered = [t.lower() for t in texts]
            results = self._hf_tok.encode_batch(lowered)
            out = []
            for enc in results:
                ids = [self._hf_to_our.get(hf_id, 0) for hf_id in enc.ids]
                if add_special:
                    ids = [self.BOS_ID] + ids + [self.EOS_ID]
                out.append(ids)
            return out
        return [self.encode(t, add_special=add_special) for t in texts]

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        special = {self.PAD_ID, self.BOS_ID, self.EOS_ID}
        b = b""
        for i in ids:
            if skip_special and i in special:
                continue
            b += self.vocab.get(i, b"?")
        return b.decode("utf-8", errors="replace")

    @property
    def pad_id(self) -> int:
        return self.PAD_ID

    def save(self, path: str) -> None:
        data = {
            "vocab_size": self.vocab_size,
            "merges": [[a, b, c] for (a, b), c in self.merges.items()],
        }
        Path(path).write_text(json.dumps(data), encoding="utf-8")
        # salva HF tokenizer ao lado (para encode rápido no load)
        if hasattr(self, "_hf_tok") and self._hf_tok is not None:
            hf_path = Path(path).with_suffix(".hf.json")
            self._hf_tok.save(str(hf_path))

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        t = cls(vocab_size=data["vocab_size"])
        next_id = cls.BASE_VOCAB + cls.NUM_SPECIAL
        for a, b, c in data["merges"]:
            pair = (a, b)
            t.merges[pair] = c
            t.vocab[c] = t.vocab.get(a, b"?") + t.vocab.get(b, b"?")
            next_id = max(next_id, c + 1)
        # tenta carregar HF tokenizer para encode rápido
        hf_path = Path(path).with_suffix(".hf.json")
        t._hf_tok = None
        t._hf_to_our = {}
        if hf_path.exists():
            try:
                from tokenizers import Tokenizer as _HFTok
                t._hf_tok = _HFTok.from_file(str(hf_path))
                # reconstrói mapa hf_id -> our_id via vocab bytes
                bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
                cs = bs[:]
                n = 0
                for b in range(256):
                    if b not in bs:
                        bs.append(b)
                        cs.append(256 + n)
                        n += 1
                _bdec = {chr(c): bv for bv, c in zip(bs, cs)}
                def _s2b(s):
                    return bytes([_bdec.get(ch, ord(ch) % 256) for ch in s])
                bytes_to_our = {bytes([i]): i for i in range(cls.BASE_VOCAB)}
                for (a, b), nid in t.merges.items():
                    bytes_to_our[t.vocab[nid]] = nid
                hf_vocab = t._hf_tok.get_vocab()
                t._hf_to_our = {hid: bytes_to_our[_s2b(s)] for s, hid in hf_vocab.items() if _s2b(s) in bytes_to_our}
            except Exception:
                t._hf_tok = None
        return t


# ─── RoPE ──────────────────────────────────────────────────────────────────

def precompute_rope(head_dim: int, ctx_len: int, base: float = 10000.0, device="cpu"):
    """Pré-computa cos/sin para RoPE."""
    assert head_dim % 2 == 0
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(ctx_len, device=device).float()
    freqs = torch.outer(t, theta)               # (T, head_dim/2)
    cos = torch.cos(freqs)                      # (T, head_dim/2)
    sin = torch.sin(freqs)                      # (T, head_dim/2)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, n_heads, T, head_dim)"""
    T = x.shape[2]
    cos = cos[:T].unsqueeze(0).unsqueeze(0)    # (1, 1, T, head_dim/2)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., ::2], x[..., 1::2]        # pares e ímpares
    x_rot = torch.stack([-x2, x1], dim=-1).flatten(-2)  # rotação
    return x * torch.cat([cos, cos], dim=-1) + x_rot * torch.cat([sin, sin], dim=-1)


# ─── Dataset ───────────────────────────────────────────────────────────────

class NextWordDataset(Dataset):
    """
    Concatena todos os textos em um único stream de tokens e fatia em janelas
    de ctx_len com stride=ctx_len//2. Usa numpy int32 pra economizar RAM.
    """
    def __init__(self, texts: list[str], tokenizer: BPETokenizer, ctx_len: int = 128):
        import numpy as np
        self.ctx_len = ctx_len
        CHUNK = 5_000  # chunks menores → flush mais frequente, menos RAM por vez
        parts: list[np.ndarray] = []
        for i in range(0, len(texts), CHUNK):
            batch = texts[i : i + CHUNK]
            encoded = tokenizer.encode_batch(batch, add_special=True)
            chunk_ids: list[int] = []
            for ids in encoded:
                if len(ids) < 4:
                    continue
                chunk_ids.extend(ids)
            parts.append(np.array(chunk_ids, dtype=np.int32))
            print(f"  tokenizado {min(i + CHUNK, len(texts)):,}/{len(texts):,}", flush=True)
            del encoded, chunk_ids

        stream = np.concatenate(parts)
        del parts
        self.data = stream
        stride = ctx_len // 2
        n = max(0, len(stream) - ctx_len - 1)
        self.offsets = np.arange(0, n, stride, dtype=np.int32)
        print(f"  NextWordDataset: {len(stream):,} tokens, {len(self.offsets):,} amostras (stride={stride})", flush=True)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int):
        import numpy as np
        start = int(self.offsets[idx])
        x = torch.from_numpy(self.data[start : start + self.ctx_len].astype(np.int64))
        y = torch.from_numpy(self.data[start + 1 : start + self.ctx_len + 1].astype(np.int64))
        return x, y


class CollateFn:
    """Picklable collate — necessário para num_workers>0 no Python 3.14+."""
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        xs, ys = zip(*batch)
        if len({len(x) for x in xs}) == 1:
            X = torch.stack(list(xs))
            Y = torch.stack(list(ys))
            return X, Y
        max_len = max(len(x) for x in xs)
        X = torch.full((len(xs), max_len), self.pad_id, dtype=torch.long)
        Y = torch.full((len(ys), max_len), self.pad_id, dtype=torch.long)
        for i, (x, y) in enumerate(zip(xs, ys)):
            X[i, : len(x)] = torch.tensor(x)
            Y[i, : len(y)] = torch.tensor(y)
        return X, Y


# ─── Gemma 3 micro ─────────────────────────────────────────────────────────
# Propriedades estruturais idênticas ao Gemma 3:
#   - Alternância local(5):global(1) em todas as camadas
#   - Local: sliding window causal, RoPE base=10K
#   - Global: full causal attention, RoPE base=1M
#   - GeGLU FFN em todos os blocos (Gemma 3 usa GeLU, não SiLU)
#   - RMSNorm (Gemma usa RMSNorm, não LayerNorm)
#   - Weight tying tok_emb <-> lm_head
# Escala micro: d_model=256, n_layers=12, window=64, ctx=512

ROPE_BASE_LOCAL  = 10_000.0
ROPE_BASE_GLOBAL = 1_000_000.0
LOCAL_WINDOW     = 64    # sliding window local (Gemma: 512/1024)
GLOBAL_RATIO     = 6     # 1 global a cada N layers (Gemma: 6)
ATTN_LOGIT_CAP   = 50.0  # soft-capping p/ estabilidade (Gemma 2/3)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.w * x / rms


class GemmaAttention(nn.Module):
    """
    Atenção causal com RoPE, QK-Norm e logit soft-capping.
    is_global=False → sliding window LOCAL_WINDOW.
    is_global=True → full causal (acessa ctx inteiro).
    Usa F.scaled_dot_product_attention (FlashAttention backend) quando possível.
    """
    def __init__(self, d_model: int, n_heads: int, ctx_len: int, dropout: float, is_global: bool):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.is_global = is_global
        self.window    = ctx_len if is_global else LOCAL_WINDOW
        self.dropout   = dropout

        self.q  = nn.Linear(d_model, d_model, bias=False)
        self.k  = nn.Linear(d_model, d_model, bias=False)
        self.v  = nn.Linear(d_model, d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.resid_drop = nn.Dropout(dropout)

        # QK-Norm: estabiliza logits de atenção
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        # máscara causal base (T×T) — convertida pra float bias p/ SDPA
        causal = torch.tril(torch.ones(ctx_len, ctx_len))
        if not is_global:
            for i in range(ctx_len):
                start = max(0, i - LOCAL_WINDOW + 1)
                causal[i, :start] = 0
        attn_bias = torch.zeros(ctx_len, ctx_len)
        attn_bias.masked_fill_(causal == 0, float("-inf"))
        self.register_buffer("attn_bias", attn_bias)

        # RoPE: base diferente para local vs global
        rope_base = ROPE_BASE_GLOBAL if is_global else ROPE_BASE_LOCAL
        cos, sin = precompute_rope(self.head_dim, ctx_len, base=rope_base)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        Q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        Q = apply_rope(self.q_norm(Q), self.rope_cos, self.rope_sin)
        K = apply_rope(self.k_norm(K), self.rope_cos, self.rope_sin)

        # logit soft-capping (Gemma 2/3): tanh(att/cap)*cap antes do softmax
        # SDPA não suporta soft-capping nativo, então usamos attn_bias manual
        scale = math.sqrt(self.head_dim)
        att = (Q @ K.transpose(-2, -1)) / scale
        att = torch.tanh(att / ATTN_LOGIT_CAP) * ATTN_LOGIT_CAP

        bias = self.attn_bias[:T, :T].unsqueeze(0).unsqueeze(0)
        att = att + bias
        att = F.softmax(att, dim=-1)
        if self.training:
            att = F.dropout(att, p=self.dropout)
        y = (att @ V).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class GemmaBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ctx_len: int, dropout: float, is_global: bool):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn  = GemmaAttention(d_model, n_heads, ctx_len, dropout, is_global)
        self.norm2 = RMSNorm(d_model)
        # GeGLU — Gemma 3 usa GeLU, não SiLU
        ffn_dim = int(d_model * 8 / 3)  # Gemma usa 8/3 × d_model
        ffn_dim = (ffn_dim + 63) // 64 * 64  # arredonda p/ múltiplo de 64
        self.ff_gate = nn.Linear(d_model, ffn_dim, bias=False)
        self.ff_up   = nn.Linear(d_model, ffn_dim, bias=False)
        self.ff_down = nn.Linear(ffn_dim, d_model, bias=False)
        self.ff_drop = nn.Dropout(dropout)
        self.is_global = is_global

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        h = self.norm2(x)
        x = x + self.ff_drop(self.ff_down(F.gelu(self.ff_gate(h)) * self.ff_up(h)))
        return x


class GemmaMicro(nn.Module):
    """
    Gemma 3 micro — mesma estrutura, escala reduzida.
    Padrão de camadas: [L, L, L, L, L, G] repetido (ratio 5:1 local:global).
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int   = 256,
        n_layers: int  = 12,
        n_heads: int   = 8,
        ctx_len: int   = 512,
        dropout: float = 0.1,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.ctx_len = ctx_len
        self.gradient_checkpointing = gradient_checkpointing
        self.emb_scale = math.sqrt(d_model)
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop    = nn.Dropout(dropout)

        # padrão Gemma 3: layer i é global se (i+1) % GLOBAL_RATIO == 0
        self.blocks = nn.ModuleList([
            GemmaBlock(
                d_model, n_heads, ctx_len, dropout,
                is_global=((i + 1) % GLOBAL_RATIO == 0)
            )
            for i in range(n_layers)
        ])

        self.norm_f = RMSNorm(d_model)
        self.head   = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self._init_weights()

        # log da estrutura
        n_global = sum(1 for i in range(n_layers) if (i + 1) % GLOBAL_RATIO == 0)
        n_local  = n_layers - n_global
        print(f"  GemmaMicro: {n_layers} layers ({n_local} local window={LOCAL_WINDOW}, {n_global} global)")

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.ctx_len, f"T={T} > ctx_len={self.ctx_len}"
        x = self.drop(self.tok_emb(idx)) * self.emb_scale
        for block in self.blocks:
            if self.training and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.head(self.norm_f(x))

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list[int],
        max_new: int   = 40,
        temperature: float = 0.8,
        top_k: int     = 50,
        top_p: float   = 0.9,
    ) -> list[int]:
        self.eval()
        device = next(self.parameters()).device
        ids = prompt_ids.copy()
        eos = BPETokenizer.EOS_ID
        for _ in range(max_new):
            ctx = torch.tensor(ids[-self.ctx_len:], dtype=torch.long).unsqueeze(0).to(device)
            logits = self(ctx)[0, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[-1]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            sorted_probs[cumsum - sorted_probs > top_p] = 0.0
            sorted_probs /= sorted_probs.sum()
            next_id = sorted_idx[torch.multinomial(sorted_probs, 1)].item()
            ids.append(next_id)
            if next_id == eos:
                break
        return ids


# ─── Treino ────────────────────────────────────────────────────────────────

def load_texts(*paths: str, max_lines_per_file: int = 0) -> list[str]:
    import glob as _glob
    texts = []
    for p in paths:
        expanded = sorted(_glob.glob(p)) or ([p] if Path(p).exists() else [])
        for ep in expanded:
            if not Path(ep).exists():
                continue
            lines: list[str] = []
            with open(ep, encoding="utf-8", errors="replace") as fh:
                for raw in fh:
                    s = raw.strip()
                    if s:
                        lines.append(s)
                    if max_lines_per_file and len(lines) >= max_lines_per_file:
                        break
            texts.extend(lines)
            total = len(lines)
            cap = f" (cap={max_lines_per_file})" if max_lines_per_file else ""
            print(f"  {ep}: {total:,} linhas{cap}")
    return texts


def train(
    data_paths: list[str] = ["dataset_portugues_br.txt"],
    save_dir: str  = "gemma_micro",
    epochs: int    = 10,
    batch_size: int = 32,
    lr: float      = 6e-4,
    ctx_len: int   = 256,
    accum_steps: int = 4,
    max_lines_per_file: int = 0,  # 0 = sem limite; use ~500_000 para arquivos gigantes
    d_model: int   = 256,
    n_heads: int   = 8,
    log_every: int = 20,          # imprime progresso a cada N batches
) -> None:
    import os
    import time
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    print(f"Device: {device} | AMP: {use_amp} | Grad accum: {accum_steps} (effective batch={batch_size * accum_steps})")

    print("Carregando textos:")
    texts = load_texts(*data_paths, max_lines_per_file=max_lines_per_file)
    print(f"Total frases: {len(texts)}")

    print("Treinando BPE tokenizer:")
    tokenizer = BPETokenizer(vocab_size=4000).train(texts)
    print(f"Vocab size: {tokenizer.vocab_size}")

    dataset = NextWordDataset(texts, tokenizer, ctx_len=ctx_len)
    del texts
    print(f"Amostras de treino: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # evita deadlock fork+Rust no Colab
        pin_memory=(device == "cuda"),
        collate_fn=CollateFn(tokenizer.pad_id),
    )

    if device == "cuda":
        torch.cuda.empty_cache()

    print("Construindo GemmaMicro:")
    model = GemmaMicro(vocab_size=tokenizer.vocab_size, ctx_len=ctx_len, d_model=d_model, n_heads=n_heads).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parâmetros: {n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler(enabled=use_amp)
    steps_per_epoch = len(loader) // accum_steps
    total_steps = epochs * steps_per_epoch

    # OneCycleLR — super-convergence: lr sobe rápido até max, desce até ~0
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1e4,
    )

    Path(save_dir).mkdir(exist_ok=True)
    n_batches = len(loader)
    print(f"Iniciando treino: {n_batches} batches/epoch, {epochs} epochs, {total_steps} optimizer steps")
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
                scheduler.step()

            if batch_idx % log_every == 0:
                step_loss = loss.item() * accum_steps
                lr_now = scheduler.get_last_lr()[0] if batch_idx > 0 else lr
                elapsed_so_far = time.time() - t_epoch
                batches_done = batch_idx + 1
                eta = (elapsed_so_far / batches_done) * (n_batches - batches_done) if batches_done else 0
                print(
                    f"  ep{epoch:02d} [{batch_idx:>5}/{n_batches}] "
                    f"loss {step_loss:.4f} | lr {lr_now:.2e} | eta {eta:.0f}s",
                    flush=True,
                )

        elapsed = time.time() - t_epoch
        avg_loss = total_loss / n_batches
        ppl = math.exp(min(avg_loss, 20))
        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:02d}/{epochs} | loss {avg_loss:.4f} | ppl {ppl:.1f} | lr {lr_now:.2e} | {elapsed:.1f}s")

        if epoch % 5 == 0:
            _demo(model, tokenizer)

        torch.save({
            "model": model.state_dict(),
            "ctx_len": ctx_len,
            "d_model": d_model,
            "n_layers": len(model.blocks),
            "n_heads": n_heads,
            "epoch": epoch,
            "loss": avg_loss,
        }, f"{save_dir}/model.pt")
        tokenizer.save(f"{save_dir}/tokenizer.json")
        print(f"  checkpoint epoch {epoch} salvo", flush=True)

    total_time = time.time() - t_start
    print(f"\nTreino completo em {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"Modelo salvo em {save_dir}/")


def _demo(model: GemmaMicro, tokenizer: BPETokenizer) -> None:
    # demo LM livre
    lm_prompts = [
        "a inteligência artificial está",
        "o brasil foi colonizado",
        "os cientistas descobriram que",
    ]
    model.eval()
    for p in lm_prompts:
        ids = tokenizer.encode(p, add_special=False)
        out = model.generate(ids, max_new=20, temperature=0.7, top_p=0.9)
        print(f"  LM [{p}] → {tokenizer.decode(out)}")



def load_model(save_dir: str = "gemma_micro") -> tuple[GemmaMicro, BPETokenizer]:
    tokenizer = BPETokenizer.load(f"{save_dir}/tokenizer.json")
    ckpt = torch.load(f"{save_dir}/model.pt", map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        cfg = {k: ckpt[k] for k in ("ctx_len", "d_model", "n_layers", "n_heads") if k in ckpt}
        model = GemmaMicro(vocab_size=tokenizer.vocab_size, **cfg)
        model.load_state_dict(ckpt["model"])
    else:
        model = GemmaMicro(vocab_size=tokenizer.vocab_size)
        model.load_state_dict(ckpt)
    return model, tokenizer


if __name__ == "__main__":
    train(
        data_paths=[
            "dataset_portugues_br.txt",
            "frases_treinamento.txt",
            "dataset_v2.txt",
            "dataset_v3.txt",
            "dataset_wiki_pt.txt",
            "wiki_hf_part_01.txt",
            "wiki_hf_part_02.txt",
            "wiki_hf_part_03.txt",
            "wiki_hf_part_07.txt",
            "wiki_hf_part_10.txt",
            "wiki_hf_part_12.txt",
        ],
    )
