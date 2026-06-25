"""Baixa Wikipedia PT do HuggingFace e salva como dataset_wiki_hf.txt (1 parágrafo/linha)."""

import re
from datasets import load_dataset
from pathlib import Path

OUTPUT = "dataset_wiki_hf.txt"
MIN_CHARS = 50
MIN_WORDS = 7
MAX_WORDS = 120
ALPHA_MIN = 0.55

_NOISE = re.compile(
    r"(cookie|javascript|clique aqui|leia mais|compartilh|assine|newsletter"
    r"|publicidade|©|direitos reservados|\[\d+\]|Ver também"
    r"|Ligações externas|Referências|Notas|categoria:|Categoria:)",
    re.IGNORECASE,
)

_ENG = re.compile(
    r"\b(the|and|of|in|to|is|was|are|that|this|with|for|on|at|by|from"
    r"|which|have|been|their|they|its|also|it|an|as|or|but|not)\b"
)

_INLINE_CLEAN = [
    (re.compile(r"\[\d+\]"), ""),
    (re.compile(r"\s{2,}"), " "),
]


def is_good(text: str) -> bool:
    if len(text) < MIN_CHARS:
        return False
    words = text.split()
    if len(words) < MIN_WORDS or len(words) > MAX_WORDS:
        return False
    alpha = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha < ALPHA_MIN:
        return False
    if _NOISE.search(text):
        return False
    eng_hits = len(_ENG.findall(text))
    if eng_hits >= 5:
        return False
    return True


def clean(text: str) -> str:
    for pat, repl in _INLINE_CLEAN:
        text = pat.sub(repl, text).strip()
    return text


def main():
    print("Baixando TucanoBR/wikipedia-PT do HuggingFace...")
    ds = load_dataset("TucanoBR/wikipedia-PT", split="train")
    print(f"Total artigos: {len(ds):,}")

    out = Path(OUTPUT)
    seen: set[str] = set()
    count = 0
    skipped = 0

    with out.open("w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            text = row["text"]
            paragraphs = re.split(r"\n+", text)

            for p in paragraphs:
                p = clean(p.strip())
                if not p or not is_good(p):
                    skipped += 1
                    continue
                if p in seen:
                    skipped += 1
                    continue
                seen.add(p)
                f.write(p + "\n")
                count += 1

            if (i + 1) % 50000 == 0:
                print(f"  {i+1:,} artigos processados | {count:,} parágrafos salvos | {skipped:,} descartados")

    print(f"\nConcluído: {count:,} parágrafos em {OUTPUT}")
    print(f"Descartados: {skipped:,}")
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"Tamanho: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
