## You need to agree to share your contact information to access this dataset

Our team may take a while to process your request

[Log in](https://huggingface.co/login?next=/datasets/soberania/jabuticaba) or [Sign Up](https://huggingface.co/join?next=/datasets/soberania/jabuticaba) to review the conditions and access this dataset content.

# Jabuticaba Corpus

### Dataset Summary

#### \[In Portuguese\]

O dataset Jabuticaba, desenvolvido pela SoberanIA, é o mais extenso corpus de língua portuguesa para Large Language Models (LLMs),
com 669 GB e mais de 139 bilhões de tokens (contagem de tokens usando o tokenizador [tiktoken](https://github.com/openai/tiktoken)
da OpenAI), contendo palavras limpas e deduplicadas prontas para uso, inclusive comercial. Este artigo detalha a rigorosa metodologia
de construção do dataset, abrangendo detecção de idioma, filtragem de conteúdo e qualidade, remoção de toxicidade, normalização,
deduplicação e tokenização. Embora seja gratuito para acesso, exige-se o preenchimento de um formulário para aprovação do acesso,
e os recursos estão disponíveis no Hugging Face, servindo como uma referência abrangente para a comunidade de pesquisa e indústria.

#### \[In English\]

The Jabuticaba dataset, developed by SoberanIA, stands as the most extensive Portuguese language corpus for Large Language Models (LLMs),
boasting 669 GB and over 139 billion clean, deduplicated tokens (token count using OpenAI's [tiktoken](https://github.com/openai/tiktoken)
tokenizer), readily available for both academic and commercial applications. This paper meticulously details the rigorous methodological
pipeline employed in its construction, encompassing language detection, content and quality filtering, toxicity removal, normalization,
deduplication, and tokenization. While free to access, a form will have to be filled out for access approval, and the resources
are available on Hugging Face, serving as a comprehensive reference for the research and industry community.

DOI: [https://doi.org/10.1590/SciELOPreprints.12696](https://doi.org/10.1590/SciELOPreprints.12696)

### Languages

The dataset is primarily in Brazilian Portuguese, but it may also contain varieties from European and African Portuguese.

## Dataset Structure

- 90 billion words
- 175 million lines
- ~3.5k JSON Lines (.jsonl) files with up to 200MB each
- 669 GB total

Token count: ~139B ( [tiktoken](https://github.com/openai/tiktoken)). Token count usually varies from tokenizer to tokenizer.

### Data Instances

```json
{
    "source": "ALC4",
    "domain": "Internet / Web scraping",
    "text": "Chevrolet Equinox\nAproveite que o novo carro SUV esportivo da Chevrolet está à venda na concessionária Palazzo.",
    "char_count": 112,
    "word_count": 17
}
```

### Data Fields

| Key/Field | Type | Note |
| --- | :-: | --- |
| `source` | str | The Identifier of the original corpus |
| `domain` | str | Domain type defined in the _Domain taxonomy_. |
| `text` | str | The actual text |
| `char_count` | int | Character count using `len(str)` |
| `word_count` | int | Word count using `str.split()` |

## Dataset Creation

### Curation Rationale

The previously compiled corpora underwent a comprehensive pipeline that included several key steps to ensure data quality.
These steps were: **Content Validation** (including toxicity validation and language detection), **Quality Filtering**,
**Document Deduplication**, and **Text Normalization**.

### Source Data

#### Initial Data Collection and Normalization

The domain taxonomy for the Jabuticaba dataset is organized into five major classes: Arts, Documents, Internet, Media, and Research:

1. **Arts / Literature**: books, poetry, novels.
2. **Documents / Academic**: papers, journals, dissertations, conference proceedings.
3. **Documents / Legal**: contracts, legislation, court rulings.
4. **Internet / Informational**: blog pages, wiki pages, how-to guides.
5. **Internet / User-Generated Content**: forum posts, personal blogs, reviews, opinion pieces.
6. **Internet / Web scraping**: varied content scraped from the internet.
7. **Research / NER**: manual data created for the NER task.

#### Who are the source language producers?

This corpus is an aggregation of multiple public datasets. Find below the complete list of corpora contained in Jabuticaba alongside other information.

| Name | Identifier | Domain | Words | Percentage | License |
| --- | --- | --- | --- | --- | --- |
| C4 | ALC4 | Internet / Web scraping | 68,238,423,781 | 75.6% | [https://huggingface.co/datasets/allenai/c4](https://huggingface.co/datasets/allenai/c4) |
| Oscar | OSCR | Internet / Web scraping | 14,173,696,374 | 15.7% | [https://oscar-project.org/#license](https://oscar-project.org/#license) |
| Opus | OPUS | Internet / Web scraping | 5,705,375,172 | 6.3% | [https://opus.nlpl.eu/](https://opus.nlpl.eu/) |
| Blogset | BGST | Internet / User-Generated Content | 1,525,281,151 | 1.7% | [https://www.inf.pucrs.br/linatural/wordpress/recursos-e-ferramentas/blogset-br/](https://www.inf.pucrs.br/linatural/wordpress/recursos-e-ferramentas/blogset-br/) |
| Wikipedia PT Dump | WKPT | Internet / Informational | 307,431,718 | 0.34% | [https://dumps.wikimedia.org/legal.html](https://dumps.wikimedia.org/legal.html) |
| XLent | XLNT | Research / NER | 207,888,796 | 0.23% | [https://data.statmt.org/xlent/](https://data.statmt.org/xlent/) |
| Acórdãos STF | ASTF | Documents / Legal | 76,388,660 | 0.08% | [https://www.inf.ufpr.br/didonet/articles/2019\_dsw\_Iudicium\_Textum\_Dataset.pdf](https://www.inf.ufpr.br/didonet/articles/2019_dsw_Iudicium_Textum_Dataset.pdf) |
| Brazilian Legal Proceedings | BRLP | Documents / Legal | 27,026,247 | 0.03% | [https://www.kaggle.com/datasets/felipepolo/brazilian-legal-proceedings](https://www.kaggle.com/datasets/felipepolo/brazilian-legal-proceedings) |
| Gutenberg Project PT | GPPT | Arts / Literature | 16,928,363 | 0.02% | [https://www.gutenberg.org/browse/languages/pt](https://www.gutenberg.org/browse/languages/pt) |
| Wikibooks | WKBK | Arts / Literature | 7,190,289 | 0.008% | [https://www.kaggle.com/datasets/dhruvildave/wikibooks-dataset](https://www.kaggle.com/datasets/dhruvildave/wikibooks-dataset) |
| Brazilian Portuguese Literature | BRLT | Arts / Literature | 3,370,103 | 0.004% | [https://www.kaggle.com/datasets/rtatman/brazilian-portuguese-literature-corpus](https://www.kaggle.com/datasets/rtatman/brazilian-portuguese-literature-corpus) |
| How2 | HOW2 | Internet / Informational | 3,018,834 | 0.003% | [https://srvk.github.io/how2-dataset/](https://srvk.github.io/how2-dataset/) |
| Fernando Pessoa | FEPE | Arts / Literature | 808,530 | 0.001% | [http://arquivopessoa.net/info/ficha](http://arquivopessoa.net/info/ficha) |
| CorpusTCC | CTCC | Documents / Academic | 52,223 | 0.00006% | [http://www.nilc.icmc.usp.br/nilc/index.php/tools-and-resources](http://www.nilc.icmc.usp.br/nilc/index.php/tools-and-resources) |
| OpiSums | OPSU | Internet / User-Generated Content | 6,328 | 0.00001% | [http://www.nilc.icmc.usp.br/nilc/index.php/tools-and-resources](http://www.nilc.icmc.usp.br/nilc/index.php/tools-and-resources) |
| Total | - | - | 90,292,886,569 | 100% | - |

## Considerations for Using the Data

### Direct Use

The Jabuticaba dataset is intended for pre-training Large Language Models. It is suitable for commercial use.
The resulting models can be used for a variety of natural language processing tasks, including:

- Conversational AI.
- Language translation.
- Text generation.

### Social Impact of Dataset

The Jabuticaba corpus holds significant potential for advancing Portuguese language models, especially for LLMs aimed at Brazilian
(and ultimately Portuguese-language-aimed) markets. By providing access to a large, clean, and deduplicated corpus, this dataset can
foster innovation in a myriad of applications, such as conversational AI, language translation, and text generation, benefiting industries,
education, and technology development.

However, it is essential to consider the societal consequences of large-scale language models trained on such data. While there is the
opportunity to improve access to information and resources in Portuguese, there is also a risk of amplifying existing biases present in
the source texts (mind mostly domain and corpus distribution).

### Discussion of Biases

Given that the Jabuticaba corpus is composed of diverse datasets scraped from various domains such as news, legal documents, user-generated
content, and academic papers, it may inherit inherent biases from these sources. Biases related to gender, race, socio-economic status, and
regional disparities may be reflected in the data. For instance, media sources might prioritize certain viewpoints, and user-generated content
may display social biases prevalent in the contributing population.

Mitigating such biases is an ongoing challenge. Users of this dataset should be aware of the potential for unintentional reinforcement of
stereotypes and discriminatory language, particularly in tasks such as sentiment analysis or text generation. Applying bias detection methods
and fairness evaluation tools is recommended when making use of this corpus for model training.

### Other Known Limitations

While Jabuticaba aims to provide the largest Portuguese corpus for LLMs, several limitations must be acknowledged. First, the dataset’s
reliance on publicly available sources could lead to incomplete coverage of certain language varieties, including regional dialects and
informal speech prevalent in oral communication, which would be key for Sociolinguistic goals. Regional varieties of register variation
(oral transcription or originally written data) may not be easily identified, as entries are not labeled as such.

Moreover, deduplication efforts, while extensive, may not eliminate all redundant content, and some noise from web-scraped data could persist.
Researchers and developers should also consider the variability in tokenization results depending on the tokenizer used, as different models
may produce varying token counts.

Lastly, the corpus predominantly reflects written language, and its applicability to tasks involving spoken language may be constrained.

## Additional Information

### Dataset Curators

- **Curated by**: Marcellus Amadeus, José Roberto Homeli da Silva, William Cruz, Rodrigo Scotti.
- **Funded by**: Piauí Institute of Technology (PIT) and the Piauí government.

### Article Citation

**BibTeX:**

```
@article{Amadeus_Cruz Castaneda_Homeli da Silva_Scotti_2025,
    title={Jabuticaba: The largest commercial corpus for LLMs in Portuguese},
    url={https://preprints.scielo.org/index.php/scielo/preprint/view/12696},
    DOI={10.1590/SciELOPreprints.12696},
    journal={SciELO Preprints},
    author={Amadeus, Marcellus and Cruz Castaneda, William Alberto and Homeli da Silva, José Roberto and Scotti, Rodrigo},
    year={2025},
    month={ago.}
}
```

**APA:**
Amadeus, M., Cruz Castaneda, W. A., Homeli da Silva, J. R., & Scotti, R. (2025). Jabuticaba: The largest commercial corpus for LLMs in Portuguese. Em SciELO Preprints. [https://doi.org/10.1590/SciELOPreprints.12696](https://doi.org/10.1590/SciELOPreprints.12696)

Copy to bucket new

Downloads last month

39

Total file size:

670 GB