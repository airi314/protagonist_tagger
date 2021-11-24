#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
set -e

conda activate protagonist
for model in en_core_web_sm en_core_web_md en_core_web_lg; do
  python -m tool.scripts.test_ner_model data/novels_titles/single_title.txt data/experiments experiments/personal_titles/ner/spacy__${model}/ spacy ${model}
done

python -m tool.scripts.test_ner_model data/novels_titles/single_title.txt data/experiments experiments/personal_titles/ner/nltk/ nltk
python -m tool.scripts.test_ner_model data/novels_titles/single_title.txt data/experiments experiments/personal_titles/ner/stanza/ stanza

conda activate protagonist2
for model in ner ner-large ner-fast; do
  python -m tool.scripts.test_ner_model data/novels_titles/single_title.txt data/experiments experiments/personal_titles/ner/flair__${model}/ flair ${model}
done

