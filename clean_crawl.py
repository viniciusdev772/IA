"""Limpeza profunda do dataset_crawl.txt — remove ruído, inglês, refs, código."""

import re
from pathlib import Path

INPUT  = "dataset_crawl.txt"
OUTPUT = "dataset_crawl.txt"

# ── detectores de língua ──────────────────────────────────────────────────────
_ENG = re.compile(
    r"\b(the|and|of|in|to|is|was|are|that|this|with|for|on|at|by|from"
    r"|which|have|been|their|they|its|also|it|an|as|or|but|not|we|he|she"
    r"|his|her|our|you|your|can|will|would|could|should|may|might|shall"
    r"|has|had|were|be|do|does|did|more|most|than|there|when|where|who"
    r"|what|how|all|any|each|both|few|more|other|into|through|during)\b"
)
_PT = re.compile(
    r"\b(que|de|do|da|em|no|na|um|uma|para|com|por|se|não|mas|como"
    r"|ao|dos|das|os|as|é|são|foi|ser|ter|uma|este|esta|esse|essa"
    r"|seu|sua|mais|também|quando|onde|porque|qual|quais|entre|sobre"
    r"|após|antes|durante|sendo|tendo|havia|estava|foram|seria)\b",
    re.IGNORECASE,
)

# ── padrões de descarte ───────────────────────────────────────────────────────
_DROP = [
    re.compile(r"^\s*↑"),                                   # refs Wikipedia ↑
    re.compile(r"^\s*«"),                                   # citações «»
    re.compile(r"^\s*[A-Z][a-z]+,\s+[A-Z][\.\s]"),        # "Watson, J."
    re.compile(r"\bpp\s+\d+"),                             # "pp 1-2"
    re.compile(r"ISBN|ISSN|DOI|doi:"),                     # refs bibliográficas
    re.compile(r"University Press|Oxford|Cambridge|Chicago, Illinois"),
    re.compile(r"^\s*\d{4}[,\.]"),                        # ano no início "2013, ..."
    re.compile(r"Ver (também|artigo|seção)|Referências|Ligações externas|Notas", re.IGNORECASE),
    re.compile(r"\[editar\s*\|", re.IGNORECASE),
    re.compile(r"^editar[A-Z]"),                           # "editarEm", "editarA"
    re.compile(r"caracteres?\s+[{}\[\]<>|]"),             # "caractere {"
    re.compile(r"[{}\[\]]{1}.*[{}\[\]]{1}"),              # qualquer { } [ ]
    re.compile(r"[{}<>\[\]|]{2,}"),                        # código/markup residual
    re.compile(r"^\s*[-–•·*]\s*$"),                       # bullets vazios
    re.compile(r"https?://|www\."),                        # URLs
    re.compile(r"@\w+\.\w+"),                              # emails
    re.compile(r"Copyright|Todos os direitos|©", re.IGNORECASE),
    re.compile(r"^\s*#|;\s*$|/\*|\*/"),                   # código
    re.compile(r"January|February|March|April|May|June|July|August"
               r"|September|October|November|December", re.IGNORECASE),  # meses em inglês
    re.compile(r"Arquivado do original|arquivado em", re.IGNORECASE),   # artefatos de ref
    re.compile(r"Dispõe sobre|Lei n[oº°]|Decreto n[oº°]|Portaria n[oº°]", re.IGNORECASE),  # legislação
    re.compile(r"\(PDF\)|\(XLS\)|\(DOC\)", re.IGNORECASE),             # links de arquivo
    re.compile(r"^\s*[a-z]\)"),                            # itens "a) b) c)"
    re.compile(r"acesse (em|o|a)|clique|baixe|download", re.IGNORECASE),
    re.compile(r"\d+\s*de\s*(janeiro|fevereiro|março|abril|maio|junho|julho"
               r"|agosto|setembro|outubro|novembro|dezembro)\s+de\s+\d{4}", re.IGNORECASE),  # datas isoladas
]

# ── limpezas inline ───────────────────────────────────────────────────────────
_INLINE = [
    (re.compile(r"\[\d+\]"),          ""),     # [1] [2]
    (re.compile(r"\([\w\s,]+\.\s*\d{4}\)"), ""),  # (Darwin, 1859)
    (re.compile(r"\s{2,}"),           " "),    # espaços duplos
    (re.compile(r"^\s*[-–•]\s*"),     ""),     # bullets início
]

MIN_CHARS  = 50
MIN_WORDS  = 7
MAX_WORDS  = 80
ALPHA_MIN  = 0.60
PT_WORDS_MIN = 2   # mínimo de palavras PT comuns


def is_english(line: str) -> bool:
    eng_hits = len(_ENG.findall(line))
    pt_hits  = len(_PT.findall(line))
    if eng_hits == 0:
        return False
    # inglês se tem mais palavras EN que PT e pelo menos 3 EN
    return eng_hits >= 3 and eng_hits > pt_hits * 1.5


def clean_line(line: str) -> str | None:
    line = line.strip()
    if not line:
        return None

    # descarte direto
    for pat in _DROP:
        if pat.search(line):
            return None

    # descarte inglês
    if is_english(line):
        return None

    # limpeza inline
    for pat, repl in _INLINE:
        line = pat.sub(repl, line).strip()

    # revalida métricas
    if len(line) < MIN_CHARS:
        return None
    words = line.split()
    if len(words) < MIN_WORDS or len(words) > MAX_WORDS:
        return None
    alpha = sum(c.isalpha() for c in line) / max(len(line), 1)
    if alpha < ALPHA_MIN:
        return None

    # mínimo de palavras PT comuns
    pt_hits = len(_PT.findall(line))
    if pt_hits < PT_WORDS_MIN:
        return None

    # encoding corrompido (ex: Ã´ Ã¢ Ã§)
    if "Ã" in line or "�" in line:
        return None

    return line


def main():
    raw = Path(INPUT).read_text(encoding="utf-8").splitlines()
    print(f"Linhas brutas:   {len(raw):,}")

    seen: set[str] = set()
    cleaned: list[str] = []
    stats = {"inglês": 0, "padrão": 0, "métricas": 0, "dup": 0}

    for line in raw:
        line = line.strip()
        if not line:
            continue

        # descarte inglês (antes de clean_line para contabilizar)
        if is_english(line):
            stats["inglês"] += 1
            continue

        result = clean_line(line)
        if result is None:
            stats["padrão"] += 1
            continue
        if result in seen:
            stats["dup"] += 1
            continue

        seen.add(result)
        cleaned.append(result)

    Path(OUTPUT).write_text("\n".join(cleaned) + "\n", encoding="utf-8")

    total_drop = sum(stats.values())
    print(f"Removidas:       {total_drop:,}")
    for k, v in stats.items():
        print(f"  {k:<12}: {v:,}")
    print(f"Restantes:       {len(cleaned):,}")
    print(f"Salvo:           {OUTPUT}")

    # amostra final
    print("\nAmostra (10 linhas):")
    import random
    for l in random.sample(cleaned, min(10, len(cleaned))):
        print(f"  {l[:100]}")


if __name__ == "__main__":
    main()
