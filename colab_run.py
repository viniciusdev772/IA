"""
Retreino do GemmaMicro no Google Colab (T4/A100).

Uso no Colab:
    !git clone https://github.com/viniciusdev772/IA.git && cd IA
    !pip install torch tokenizers requests trafilatura -q
    exec(open('colab_run.py').read())
"""
import os, sys, glob, shutil

# ── 1. Monta Google Drive ──────────────────────────────────────────────────
from google.colab import drive
drive.mount("/content/drive")

DRIVE_DIR = "/content/drive/MyDrive/GemmaMicro"
os.makedirs(DRIVE_DIR, exist_ok=True)

# Testa escrita
_test = f"{DRIVE_DIR}/_write_test.txt"
open(_test, "w").write("ok")
assert os.path.exists(_test), "ERRO: Drive sem permissão de escrita"
os.remove(_test)
print("Drive OK")

# ── 2. Resolve datasets ────────────────────────────────────────────────────
CONTENT = "/content/IA"  # repo clonado aqui

# Datasets pequenos já no repo
SMALL = [
    "dataset_portugues_br.txt",
    "frases_treinamento.txt",
    "dataset_v2.txt",
    "dataset_v3.txt",
    "dataset_wiki_pt.txt",
]

# Datasets grandes: tenta Drive primeiro, senão parte do repo/colab
BIG_CANDIDATES = [
    f"{DRIVE_DIR}/dataset_crawl.txt",
    f"{CONTENT}/dataset_crawl.txt",
    "/content/dataset_crawl.txt",
]

data_paths = [f"{CONTENT}/{f}" for f in SMALL if os.path.exists(f"{CONTENT}/{f}")]

# Adiciona dataset_crawl se disponível
for c in BIG_CANDIDATES:
    if os.path.exists(c):
        data_paths.append(c)
        print(f"dataset_crawl: {c}")
        break

# Junta partes wiki_hf se existirem (Drive ou /content)
wiki_hf_out = "/content/dataset_wiki_hf.txt"
if not os.path.exists(wiki_hf_out):
    parts = sorted(glob.glob(f"{DRIVE_DIR}/wiki_hf_part_*.txt"))
    if not parts:
        parts = sorted(glob.glob("/content/wiki_hf_part_*.txt"))
    if not parts:
        parts = sorted(glob.glob(f"{CONTENT}/wiki_hf_part_*.txt"))
    if parts:
        print(f"Juntando {len(parts)} partes wiki_hf...")
        with open(wiki_hf_out, "w", encoding="utf-8") as out:
            for p in parts:
                with open(p, encoding="utf-8", errors="replace") as inp:
                    out.write(inp.read())
        print(f"dataset_wiki_hf.txt criado ({os.path.getsize(wiki_hf_out)//1024//1024} MB)")

if os.path.exists(wiki_hf_out):
    data_paths.append(wiki_hf_out)

print(f"\nDatasets ({len(data_paths)}):")
for p in data_paths:
    mb = os.path.getsize(p) / 1024 / 1024
    print(f"  {p}  ({mb:.1f} MB)")

if not data_paths:
    sys.exit("ERRO: nenhum dataset encontrado")

# ── 3. Treina ──────────────────────────────────────────────────────────────
sys.path.insert(0, CONTENT)
import nn

nn.train(
    data_paths=data_paths,
    save_dir=DRIVE_DIR,
    epochs=20,
    batch_size=128,
    lr=6e-4,
    ctx_len=512,
    accum_steps=4,
)

# ── 4. Resumo ──────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"TREINO COMPLETO — salvo em: {DRIVE_DIR}")
for f in sorted(os.listdir(DRIVE_DIR)):
    mb = os.path.getsize(f"{DRIVE_DIR}/{f}") / 1024 / 1024
    print(f"  {f}: {mb:.1f} MB")
print(f"{'='*60}")
