Dataset Viewer

[Auto-converted to Parquet](https://huggingface.co/datasets/TucanoBR/wikipedia-PT/tree/refs%2Fconvert%2Fparquet/default) APIEmbed [Duplicate](https://huggingface.co/datasets/TucanoBR/wikipedia-PT?duplicate=true)Data Studio

Subset (1)
default·1.1M rows
default (1.1M rows)
Split (1)

train·1.1M rows
train (1.1M rows)

SQL
Console

| text<br>stringlengths<br>6<br>443k |
| --- |
| Astronomia é uma ciência natural que estuda corpos celestes (como estrelas, planetas, cometas, nebulosas, aglomerados de estrelas, galáxias) e fenômenos que se originam fora da atmosfera da Terra (como a radiação cósmica de fundo em micro-ondas). Preocupada com a evolução, a física e a química de objetos celestes, bem ... |
| Abel — personagem bíblico<br>Abel (filme) — filme de 1986<br>Prémio Abel — prémio dado a matemáticos em homenagem a Niels Henrik Abel<br>Pessoas<br>Karl Friedrich Abel — compositor alemão<br>Niels Henrik Abel — matemático norueguês<br>Thomas Abel — padre e mártir inglês<br>Abel Ferreira (treinador) — treinador português<br>Gottlieb Fr... |
| Alexandre I da Escócia — rei da Escócia (r. 6/12/1214–1249)<br>Alexandre III da Escócia — rei da Escócia (r. 1249–/03/1286)<br>Desambiguações de antropônimos<br>Desambiguações de história |
| Aves são uma classe de seres vivos vertebrados endotérmicos caracterizada pela presença de penas, um bico sem dentes, oviparidade de casca rígida, elevado metabolismo, um coração com quatro câmaras e um esqueleto pneumático resistente e leve. As aves estão presentes em todas as regiões do mundo e variam significativame... |
| Aldous Leonard Huxley (Godalming, 26 de julho de 1894 — Los Angeles, 22 de novembro de 1963) foi um escritor inglês e um dos mais proeminentes membros da família Huxley. Mais conhecido pelos seus romances, como Admirável Mundo Novo e diversos ensaios, Huxley também editou a revista Oxford Poetry e publicou contos, poes... |
| "Amazonas é uma das 27 unidades federativas do Brasil. Está situado na Região Norte, sendo o maio(...TRUNCATED) |
| "Antoine-Henri Becquerel (Paris, — Le Croisic, ) foi um físico francês. Becquerel foi o respons(...TRUNCATED) |
| "André-Marie Ampère (Lyon, — Marselha, ) foi um físico, filósofo, cientista e matemático fra(...TRUNCATED) |
| "Aruaques, também conhecidos como aravaques e arauaques, são numerosos grupos indígenas da Améri(...TRUNCATED) |
| "Abolicionismo — movimento político que visava o fim da escravidão\\nAbolição (Rio de Janeiro) (...TRUNCATED) |

End of preview. [Expand in Data Studio](https://huggingface.co/datasets/TucanoBR/wikipedia-PT/viewer/default/train)

* * *

# Wikipedia-PT

## Dataset Summary

The Portuguese portion of the [Wikipedia dataset](https://huggingface.co/datasets/wikimedia/wikipedia).

### Supported Tasks and Leaderboards

The dataset is generally used for Language Modeling.

### Languages

Portuguese

## Dataset Structure

### Data Instances

An example looks as follows:

```
{
 'text': 'Abril é o quarto mês...'
}
```

### Data Fields

- `text` (`str`): Text content of the article.

### Data Splits

All configurations contain a single `train` split.

## Dataset Creation

### Initial Data Collection and Normalization

The dataset is built from the Wikipedia dumps: [https://dumps.wikimedia.org](https://dumps.wikimedia.org/)

You can find the full list of languages and dates here: [https://dumps.wikimedia.org/backup-index.html](https://dumps.wikimedia.org/backup-index.html)

The articles have been parsed using the [`mwparserfromhell`](https://mwparserfromhell.readthedocs.io/) tool.

When uploading the data files for the 20231101 dump, we noticed that the Wikimedia Dumps website does not contain this date dump
for the "bbc", "dga", nor "zgh" Wikipedias. We have reported the issue to the Wikimedia Phabricator: [https://phabricator.wikimedia.org/T351761](https://phabricator.wikimedia.org/T351761)

### Licensing Information

Copyright licensing information: [https://dumps.wikimedia.org/legal.html](https://dumps.wikimedia.org/legal.html)

All original textual content is licensed under the [GNU Free Documentation License](https://www.gnu.org/licenses/fdl-1.3.html) (GFDL)
and the [Creative Commons Attribution-Share-Alike 3.0 License](https://creativecommons.org/licenses/by-sa/3.0/).
Some text may be available only under the Creative Commons license; see their [Terms of Use](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use) for details.
Text written by some authors may be released under additional licenses or into the public domain.

## Citation Information

```
@ONLINE{wikidump,
    author = "Wikimedia Foundation",
    title  = "Wikimedia Downloads",
    url    = "https://dumps.wikimedia.org"
}
```

Copy to bucket new

Use this dataset

Downloads last month

232

Number of rows:

1,103,446

Total file size:

1.69 GB

## Models trained or fine-tuned on TucanoBR/wikipedia-PT

[Text Generation • 96M•Updated Jan 27• 21](https://huggingface.co/tensorblock/Felladrin_Minueza-2-96M-GGUF)

[Text Generation • 96M•Updated Apr 5, 2025• 15• 6](https://huggingface.co/Felladrin/Minueza-2-96M)