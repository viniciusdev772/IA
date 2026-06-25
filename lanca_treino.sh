#!/bin/bash
set -e

cd /var/home/vinicius/IA

echo "=== VdevLabs 1 — Lançador de Treino Colab ==="
echo ""

# 1. Cria sessão GPU
echo "[1/6] Criando sessão GPU T4..."
colab new -s treino3 --gpu T4
echo ""

# 2. Monta Google Drive
echo "[2/6] Montando Google Drive..."
colab drivemount -s treino3
echo ""

# 3. Upload scripts
echo "[3/6] Enviando scripts..."
colab upload -s treino3 nn.py nn.py
colab upload -s treino3 colab_run.py colab_run.py
echo ""

# 4. Upload datasets pequenos
echo "[4/6] Enviando datasets..."
for f in dataset_portugues_br.txt frases_treinamento.txt dataset_v2.txt dataset_v3.txt dataset_wiki_pt.txt dataset_crawl.txt; do
    if [ -f "$f" ]; then
        echo "  -> $f"
        colab upload -s treino3 "$f" "$f"
    else
        echo "  AVISO: $f não encontrado, pulando"
    fi
done
echo ""

# 5. Upload wiki_hf em partes (todas ≤10MB)
echo "[5/6] Enviando Wikipedia PT em partes..."
for f in wiki_hf_part_*.txt; do
    if [ -f "$f" ]; then
        echo "  -> $f"
        colab upload -s treino3 "$f" "$f"
    fi
done
echo ""

# 6. Lança treino
echo "[6/6] Lançando treino (timeout 2h)..."
echo "  Modelo salvo em: Google Drive > MyDrive > vdevlabs1/"
echo "  Checkpoint a cada epoch — pode desligar PC"
echo ""
colab exec -s treino3 -f colab_run.py --timeout 7200

echo ""
echo "=== Treino finalizado ==="
echo "Modelo em: Google Drive > MyDrive > vdevlabs1/"
