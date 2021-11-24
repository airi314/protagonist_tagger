#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
set -e

conda activate protagonist
for model in en_core_web_sm en_core_web_md en_core_web_lg; do
  python -m tool.scripts.test_ner_model data/novels_titles/combined_set.txt data/testing_sets/test experiments/"$1"/ner/spacy__${model}/ spacy ${model} "${@:2}"
done

python -m tool.scripts.test_ner_model data/novels_titles/combined_set.txt data/testing_sets/test experiments/"$1"/ner/nltk/ nltk "${@:2}"
python -m tool.scripts.test_ner_model data/novels_titles/combined_set.txt data/testing_sets/test experiments/"$1"/ner/stanza/ stanza "${@:2}"

conda activate protagonist2
for model in ner ner-large ner-fast; do
  python -m tool.scripts.test_ner_model data/novels_titles/combined_set.txt data/testing_sets/test experiments/"$1"/ner/flair__${model}/ flair ${model} "${@:2}"
done

