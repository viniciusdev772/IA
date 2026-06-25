Dataset Preview

APIEmbed [Duplicate](https://huggingface.co/datasets/PleIAs/Portuguese-PD?duplicate=true)Data Studio

Subset (1)
default
default
Split (1)

train
train

The full dataset viewer is not available (click to read why). Only showing a preview of the rows.

The dataset generation failed

```
Error code:   DatasetGenerationError
Exception:    CastError
Message:      Couldn't cast
identifier: string
creator: string
title: string
publication_date: int32
language: string
language_code: string
text: string
word_count: int32
character_count: int32
-- schema metadata --
pandas: '{"index_columns": [], "column_indexes": [], "columns": [{"name":' + 1142\
to\
{'identifier': Value(dtype='string', id=None), 'creator': Value(dtype='string', id=None), 'title': Value(dtype='string', id=None), 'publication_date': Value(dtype='int64', id=None), 'word_count': Value(dtype='int64', id=None), 'text': Value(dtype='string', id=None)}\
because column names don't match\
Traceback:    Traceback (most recent call last):\
                File "/src/services/worker/src/worker/job_runners/config/parquet_and_info.py", line 1492, in compute_config_parquet_and_info_response\
                  fill_builder_info(builder, hf_endpoint=hf_endpoint, hf_token=hf_token, validate=validate)\
                File "/src/services/worker/src/worker/job_runners/config/parquet_and_info.py", line 683, in fill_builder_info\
                  ) = retry_validate_get_features_num_examples_size_and_compression_ratio(\
                File "/src/services/worker/src/worker/job_runners/config/parquet_and_info.py", line 602, in retry_validate_get_features_num_examples_size_and_compression_ratio\
                  validate(pf)\
                File "/src/services/worker/src/worker/job_runners/config/parquet_and_info.py", line 640, in validate\
                  raise TooBigRowGroupsError(\
              worker.job_runners.config.parquet_and_info.TooBigRowGroupsError: Parquet file has too big row groups. First row group has 1121326835 which exceeds the limit of 300000000\
\
              During handling of the above exception, another exception occurred:\
\
              Traceback (most recent call last):\
                File "/src/services/worker/.venv/lib/python3.9/site-packages/datasets/builder.py", line 1995, in _prepare_split_single\
                  for _, table in generator:\
                File "/src/services/worker/src/worker/job_runners/config/parquet_and_info.py", line 797, in wrapped\
                  for item in generator(*args, **kwargs):\
                File "/src/services/worker/.venv/lib/python3.9/site-packages/datasets/packaged_modules/parquet/parquet.py", line 97, in _generate_tables\
                  yield f"{file_idx}_{batch_idx}", self._cast_table(pa_table)\
                File "/src/services/worker/.venv/lib/python3.9/site-packages/datasets/packaged_modules/parquet/parquet.py", line 75, in _cast_table\
                  pa_table = table_cast(pa_table, self.info.features.arrow_schema)\
                File "/src/services/worker/.venv/lib/python3.9/site-packages/datasets/table.py", line 2302, in table_cast\
                  return cast_table_to_schema(table, schema)\
                File "/src/services/worker/.venv/lib/python3.9/site-packages/datasets/table.py", line 2256, in cast_table_to_schema\
                  raise CastError(\
              datasets.table.CastError: Couldn't cast\
              identifier: string\
              creator: string\
              title: string\
              publication_date: int32\
              language: string\
              language_code: string\
              text: string\
              word_count: int32\
              character_count: int32\
              -- schema metadata --\
              pandas: '{"index_columns": [], "column_indexes": [], "columns": [{"name":' + 1142\
              to\
              {'identifier': Value(dtype='string', id=None), 'creator': Value(dtype='string', id=None), 'title': Value(dtype='string', id=None), 'publication_date': Value(dtype='int64', id=None), 'word_count': Value(dtype='int64', id=None), 'text': Value(dtype='string', id=None)}\
              because column names don't match\
\
              The above exception was the direct cause of the following exception:\
\
              Traceback (most recent call last):\
                File "/src/services/worker/src/worker/job_runners/config/parquet_and_info.py", line 1505, in compute_config_parquet_and_info_response\
                  parquet_operations, partial, estimated_dataset_info = stream_convert_to_parquet(\
                File "/src/services/worker/src/worker/job_runners/config/parquet_and_info.py", line 1099, in stream_convert_to_parquet\
                  builder._prepare_split(\
                File "/src/services/worker/.venv/lib/python3.9/site-packages/datasets/builder.py", line 1882, in _prepare_split\
                  for job_id, done, content in self._prepare_split_single(\
                File "/src/services/worker/.venv/lib/python3.9/site-packages/datasets/builder.py", line 2038, in _prepare_split_single\
                  raise DatasetGenerationError("An error occurred while generating the dataset") from e\
              datasets.exceptions.DatasetGenerationError: An error occurred while generating the dataset\
```\
\
Need help to make the dataset viewer work? Make sure to review [how to configure the dataset viewer](https://huggingface.co/docs/hub/datasets-data-files-configuration), and [open a discussion](https://huggingface.co/datasets/PleIAs/Portuguese-PD/discussions/new?title=Dataset+Viewer+issue%3A+DatasetGenerationError&description=The+dataset+viewer+is+not+working.%0A%0AError+details%3A%0A%0A%60%60%60%0AError+code%3A+++DatasetGenerationError%0AException%3A++++CastError%0AMessage%3A++++++Couldn%27t+cast%0Aidentifier%3A+string%0Acreator%3A+string%0Atitle%3A+string%0Apublication_date%3A+int32%0Alanguage%3A+string%0Alanguage_code%3A+string%0Atext%3A+string%0Aword_count%3A+int32%0Acharacter_count%3A+int32%0A--+schema+metadata+--%0Apandas%3A+%27%7B%22index_columns%22%3A+%5B%5D%2C+%22column_indexes%22%3A+%5B%5D%2C+%22columns%22%3A+%5B%7B%22name%22%3A%27+%2B+1142%0Ato%0A%7B%27identifier%27%3A+Value%28dtype%3D%27string%27%2C+id%3DNone%29%2C+%27creator%27%3A+Value%28dtype%3D%27string%27%2C+id%3DNone%29%2C+%27title%27%3A+Value%28dtype%3D%27string%27%2C+id%3DNone%29%2C+%27publication_date%27%3A+Value%28dtype%3D%27int64%27%2C+id%3DNone%29%2C+%27word_count%27%3A+Value%28dtype%3D%27int64%27%2C+id%3DNone%29%2C+%27text%27%3A+Value%28dtype%3D%27string%27%2C+id%3DNone%29%7D%0Abecause+column+names+don%27t+match%0ATraceback%3A++++Traceback+%28most+recent+call+last%29%3A%0A++++++++++++++++File+%22%2Fsrc%2Fservices%2Fworker%2Fsrc%2Fworker%2Fjob_runners%2Fconfig%2Fparquet_and_info.py%22%2C+line+1492%2C+in+compute_config_parquet_and_info_response%0A++++++++++++++++++fill_builder_info%28builder%2C+hf_endpoint%3Dhf_endpoint%2C+hf_token%3Dhf_token%2C+validate%3Dvalidate%29%0A++++++++++++++++File+%22%2Fsrc%2Fservices%2Fworker%2Fsrc%2Fworker%2Fjob_runners%2Fconfig%2Fparquet_and_info.py%22%2C+line+683%2C+in+fill_builder_info%0A++++++++++++++++++%29+%3D+retry_validate_get_features_num_examples_size_and_compression_ratio%28%0A++++++++++++++++File+%22%2Fsrc%2Fservices%2Fworker%2Fsrc%2Fworker%2Fjob_runners%2Fconfig%2Fparquet_and_info.py%22%2C+line+602%2C+in+retry_validate_get_features_num_examples_size_and_compression_ratio%0A++++++++++++++++++validate%28pf%29%0A++++++++++++++++File+%22%2Fsrc%2Fservices%2Fworker%2Fsrc%2Fworker%2Fjob_runners%2Fconfig%2Fparquet_and_info.py%22%2C+line+640%2C+in+validate%0A++++++++++++++++++raise+TooBigRowGroupsError%28%0A++++++++++++++worker.job_runners.config.parquet_and_info.TooBigRowGroupsError%3A+Parquet+file+has+too+big+row+groups.+First+row+group+has+1121326835+which+exceeds+the+limit+of+300000000%0A++++++++++++++%0A++++++++++++++During+handling+of+the+above+exception%2C+another+exception+occurred%3A%0A++++++++++++++%0A++++++++++++++Traceback+%28most+recent+call+last%29%3A%0A++++++++++++++++File+%22%2Fsrc%2Fservices%2Fworker%2F.venv%2Flib%2Fpython3.9%2Fsite-packages%2Fdatasets%2Fbuilder.py%22%2C+line+1995%2C+in+_prepare_split_single%0A++++++++++++++++++for+_%2C+table+in+generator%3A%0A++++++++++++++++File+%22%2Fsrc%2Fservices%2Fworker%2Fsrc%2Fworker%2Fjob_runners%2Fconfig%2Fparquet_and_info.py%22%2C+line+797%2C+in+wrapped%0A++++++++++++++++++for+item+in+generator%28*args%2C+**kwargs%29%3A%0A++++++++++++++++File+%22%2Fsrc%2Fservices%2Fworker%2F.venv%2Flib%2Fpython3.9%2Fsite-packages%2Fdatasets%2Fpackaged_modules%2Fparquet%2Fparquet.py%22%2C+line+97%2C+in+_generate_tables%0A++++++++++++++++++yield+f%22%7Bfile_idx%7D_%7Bbatch_idx%7D%22%2C+self._cast_table%28pa_table%29%0A++++++++++++++++File+%22%2Fsrc%2Fservices%2Fworker%2F.venv%2Flib%2Fpython3.9%2Fsite-packages%2Fdatasets%2Fpackaged_modules%2Fparquet%2Fparquet.py%22%2C+line+75%2C+in+_cast_table%0A++++++++++++++++++pa_table+%3D+table_cast%28pa_table%2C+self.info.features.arrow_schema%29%0A++++++++++++++++File+%22%2Fsrc%2Fservices%2Fworker%2F.venv%2Flib%2Fpython3.9%2Fsite-packages%2Fdatasets%2Ftable.py%22%2C+line+2302%2C+in+table_cast%0A++++++++++++++++++return+cast_table_to_schema%28table%2C+schema%29%0A++++++++++++++++File+%22%2Fsrc%2Fservices%2Fworker%2F.venv%2Flib%2Fpython3.9%2Fsite-packages%2Fdatasets%2Ftable.py%22%2C+line+2256%2C+in+cast_table_to_schema%0A++++++++++++++++++raise+CastError%28%0A++++++++++++++datasets.table.CastError%3A+Couldn%27t+cast%0A++++++++++++++identifier%3A+string%0A++++++++++++++creator%3A+string%0A++++++++++++++title%3A+string%0A++++++++++++++publication_date%3A+int32%0A++++++++++++++language%3A+string%0A++++++++++++++language_code%3A+string%0A++++++++++++++text%3A+string%0A++++++++++++++word_count%3A+int32%0A++++++++++++++character_count%3A+int32%0A++++++++++++++--+schema+metadata+--%0A++++++++++++++pandas%3A+%27%7B%22index_columns%22%3A+%5B%5D%2C+%22column_indexes%22%3A+%5B%5D%2C+%22columns%22%3A+%5B%7B%22name%22%3A%27+%2B+1142%0A++++++++++++++to%0A++++++++++++++%7B%27identifier%27%3A+Value%28dtype%3D%27string%27%2C+id%3DNone%29%2C+%27creator%27%3A+Value%28dtype%3D%27string%27%2C+id%3DNone%29%2C+%27title%27%3A+Value%28dtype%3D%27string%27%2C+id%3DNone%29%2C+%27publication_date%27%3A+Value%28dtype%3D%27int64%27%2C+id%3DNone%29%2C+%27word_count%27%3A+Value%28dtype%3D%27int64%27%2C+id%3DNone%29%2C+%27text%27%3A+Value%28dtype%3D%27string%27%2C+id%3DNone%29%7D%0A++++++++++++++because+column+names+don%27t+match%0A++++++++++++++%0A++++++++++++++The+above+exception+was+the+direct+cause+of+the+following+exception%3A%0A++++++++++++++%0A++++++++++++++Traceback+%28most+recent+call+last%29%3A%0A++++++++++++++++File+%22%2Fsrc%2Fservices%2Fworker%2Fsrc%2Fworker%2Fjob_runners%2Fconfig%2Fparquet_and_info.py%22%2C+line+1505%2C+in+compute_config_parquet_and_info_response%0A++++++++++++++++++parquet_operations%2C+partial%2C+estimated_dataset_info+%3D+stream_convert_to_parquet%28%0A++++++++++++++++File+%22%2Fsrc%2Fservices%2Fworker%2Fsrc%2Fworker%2Fjob_runners%2Fconfig%2Fparquet_and_info.py%22%2C+line+1099%2C+in+stream_convert_to_parquet%0A++++++++++++++++++builder._prepare_split%28%0A++++++++++++++++File+%22%2Fsrc%2Fservices%2Fworker%2F.venv%2Flib%2Fpython3.9%2Fsite-packages%2Fdatasets%2Fbuilder.py%22%2C+line+1882%2C+in+_prepare_split%0A++++++++++++++++++for+job_id%2C+done%2C+content+in+self._prepare_split_single%28%0A++++++++++++++++File+%22%2Fsrc%2Fservices%2Fworker%2F.venv%2Flib%2Fpython3.9%2Fsite-packages%2Fdatasets%2Fbuilder.py%22%2C+line+2038%2C+in+_prepare_split_single%0A++++++++++++++++++raise+DatasetGenerationError%28%22An+error+occurred+while+generating+the+dataset%22%29+from+e%0A++++++++++++++datasets.exceptions.DatasetGenerationError%3A+An+error+occurred+while+generating+the+dataset%0A%60%60%60%0A%0A%0Acc+%40lhoestq+%40cfahlgren1.) for direct support.\
\
| identifier<br>string | creator<br>string | title<br>string | publication\_date<br>int64 | word\_count<br>int64 | text<br>string |\
| --- | --- | --- | --- | --- | --- |\
| osvaresillustre03silvgoog | Pereira da Silva, João Manuel, 1819?-1898. \[from old catalog\] | Os varões illustres do Brazil durante os tempos coloniaes | 1,868 | 53,834 | "Google \\n\\n\\n\\nThis is a digilal copy of a bix>k lhai was preservcd for general ions oii library sh(...TRUNCATED) |\
| djaymeoudominaod00ribe | Ribeiro, TomÃ¡s, 1831-1901 \| Castilho, Antonio Feliciano de, 1800-1875 | D. Jayme, ou, A dominaÃ§Ã£o de Castella : poema | 1,862 | 55,168 | "%ía \\n\\n\\n.'/. \\n\\n\\nf^rf. \\n\\n\\nD. JAYME \\n\\n\\nou \\n\\n\\nA DOMINAÇÃO DE CASTELLA. \\n\\nPO(...TRUNCATED) |\
| archivosdomuseu66muse | Museu Nacional (Brazil) | Archivos do Museu Nacional do Rio de Janeiro | 1,876 | 145,754 | "ISSN 0365-4508 \\n\\n\\n\\n\\n\\nNunquam aliud natura, aliud sapienta dicit \\nJuvenal, 14, 321 \\nIn silvi(...TRUNCATED) |\
| jornaldesciiasma13acad | Academia das Ciias de Lisboa | Jornal de sciias mathemcas, physicas e naturaes | 1,866 | 103,623 | "\[ :■::■■ \\n\\nr?!!í \\n\\nt \|.;; \\n\\ní \\n\\nJ'i \\n\\nFOR THE PEOPLE \\n\\nFOR EDVCATION (...TRUNCATED) |\
| terceiradecadada00barr | Barros, João de, 1496-1570 | "Terceira decada da Asia de Ioam de Barros: : dos feytos que os portugueses fizeram no descobrimento(...TRUNCATED) | 1,563 | 223,924 | "\\"V \\n\\nSC \\n\\n■>%■\_ \\n\\nr \\n\\n^. \\n\\nr; \\n\\nli\*\\" \\n\\n\\nI \\n\\n\\nV- \\n\\n\\ní \\n\\n\\n\[/•'^V \\n\(...TRUNCATED) |\
| instrucesquesere00port | Portugal. Sovereign (1816-1826 : John VI) \| Raguet, Condy. fmo RPJCB | "Instrucções a que se refere o meu real decreto de 22 de abril de 1821. : O principe real do Reino(...TRUNCATED) | 1,821 | 998 | "1 II III ji .■«III II LumíK.tmmimmfi^imm^^mmf \\n\\n\\n■ T^i,'. \\n\\n\\niÁfTL i Mlc^o ^h(...TRUNCATED) |\
| ofazendeirodobra07vell | Velloso, José Mariano da Conceição, 1742-1811 | "O fazendeiro do Brazil : melhorado na economia rural dos generos já cultivados, e de outros, que s(...TRUNCATED) | 1,798 | 93,383 | "o FAZENDEIRO \\n\\nDO B R A Z I L, \\n\\nCULTIVADOR, \\n\\n\\nOFAZENDEIRO \\n\\nDO BRAZIL, \\n\(...TRUNCATED) |\
| prcisdelhistoir02logoog | Adolphe Loève-Veimars | Précis de l'histoire de la littérature française: depuis son origine jusqu'à nos jours | 1,838 | 92,044 | "Google \\n\\n\\n\\nThis is a digital copy of a book that was prcscrvod for gcncrations on library shclv(...TRUNCATED) |\
| relatpezos1883minfz | Brasil. Ministério da Fazenda | "Relatório sobre o melhoramento do systema de pezos e medidas e monetario apresentado ao Illmo. e E(...TRUNCATED) | 1,834 | 38,893 | "é \\n\\n\\n9 \\n\\n\\n'“«'.L -V\|;OR lo£\_\| ' \\n\\n\\nW \\n\\n\\n/■ \\n\\n\\nO MELHORAMENTff D OTS YSTEMi(...TRUNCATED) |\
| arquivoaoriano01unkngoog | null | Arquivo açoriano | 1,878 | 246,635 | "This is a digital copy of a book that was preserved for generations on library shelv(...TRUNCATED) |\
\
End of preview.\
\
**YAML Metadata**\
**Warning:** empty or missing yaml metadata in repo card\
\
Check out the [documentation](https://huggingface.co/docs/hub/datasets-cards) for more information.\
\
# 🇵🇹 Portuguese Public Domain 🇵🇹\
\
**Portuguese-Public Domain** or **Portuguese-PD** is a large collection aiming to aggregate all Portuguese monographies and periodicals in the public domain. As of March 2024, it is the biggest Portuguese open corpus.\
\
## Dataset summary\
\
The collection contains 7,840 individual titles making up 672,197,538 words recovered from multiple sources, including Internet Archive and various European national libraries and cultural heritage institutions. Each parquet file has the full text of 2,000 books selected at random.\
\
## Curation method\
\
The composition of the dataset adheres to the criteria for public domain works in the EU and, consequently, all Berne-countries for EU authors: any publication whose author is dead for more than 70 years. Additionally, the initial consolidation of public domain status for cultural heritage operates in the EU under the 2019 Copyright Directive (art. 14).\
\
As of March 2024, to limit rights verification, we have retained exclusively titles published prior to 1884.\
\
The corpus will be expanded at a later stage to encompass late 19th century and early 20th century publications, after checking for public domain validity.\
\
## Uses\
\
The collection aims to expand the availability of open works for the training of Large Language Models. The text can be used for model training and republished without restriction for reproducibility purposes.\
\
The rationales for creation of this collection are multifold:\
\
- **Scientific**: We observe that the closure of training corpora represents a major barrier to AI research. Large language models face a real crisis of reproducibility.\
- **Legal**: With the adoption of the AI Act with its obligations in terms of copyright law compliance for the pretraining corpora, the European AI ecosystem will have to change its provenance practices.\
- **Cultural**: The linguistic diversity of the European Union is currently underrepresented. Unlike web archives, open, heritage, administrative, or scientific texts are often of high quality: they are long, multilingual, and editorialized publications.\
- **Economical**: Today, value capture is concentrated on players whose financial resources are already considerable, allowing them to collect or purchase data at a high price. Making a royalty-free corpus available to as many people as possible frees innovation in uses and minimizes economic dependencies on dominant actors.\
\
## License\
\
The entire collection is in the public domain in all regions. This means that the patrimonial rights of each individual or collective right holders have expired.\
\
There has been a debate for years in Europe over the definition of public domain and the possibility to restrict its use. Since 2019, the EU Copyright Directive states that "Member States shall provide that, when the term of protection of a work of visual art has expired, any material resulting from an act of reproduction of that work is not subject to copyright or related rights, unless the material resulting from that act of reproduction is original in the sense that it is the author's own intellectual creation." (art. 14)\
\
## Future work\
\
This dataset is not a one-time work but will continue to evolve significantly in three directions:\
\
- Expansion of the dataset to the late 19th and early 20th century works and its further enhancement with currently unexploited collections coming from European patrimonial data repositories.\
- Correction of computer generated errors in the text. All the texts have been transcribed automatically through the use of Optical Character Recognition (OCR) software. The original files have been digitized over a long time period (since the mid-2000s) and some documents should be. Future versions will strive either to re-OCRize the original text or use experimental LLM models for partial OCR correction.\
- Enhancement of the structure/editorial presentation of the original text. Some parts of the original documents are likely unwanted for large scale analysis or model training (header, page count…). Additionally, some advanced document structures like tables or multi-column layout are unlikely to be well-formatted.\
\
## Acknowledgements\
\
The corpus was stored and processed with the generous support of Scaleway. It was built up with the support and concerted efforts of the state start-up LANGU:IA (start-up d’Etat), supported by the French Ministry of Culture and DINUM, as part of the prefiguration of the service offering of the Alliance for Language technologies EDIC (ALT-EDIC).\
\
Corpus collection has been largely facilitated thanks to the open science LLM community insights, cooperation and support (Occiglot, Eleuther AI, OpenLLM France, Allen AI).\
\
![](https://github.com/mch-dd/datasetlogo/blob/main/scaleway.jpeg?raw=true)![](https://github.com/mch-dd/datasetlogo/blob/main/ministere.png?raw=true)![](https://github.com/mch-dd/datasetlogo/blob/main/occiglot.jpg?raw=true)\
\
Copy to bucket new\
\
Downloads last month\
\
289\
\
Total file size:\
\
3.32 GB\
\
## Collection including PleIAs/Portuguese-PD\
\
[A multilingual dataset of public domain books and newspapers.•25 items•Updated Mar 2• 135](https://huggingface.co/collections/PleIAs/openculture)