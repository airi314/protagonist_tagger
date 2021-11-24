#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
set -e

conda activate protagonist
for model in en_core_web_sm en_core_web_md en_core_web_lg; do
  python -m tool.scripts.compute_metrics data/novels_titles/combined_set.txt data/testing_sets/"$1"/ experiments/"$2"/ner/spacy__${model}/ner_model_annotated/ experiments/"$2"/ner/spacy__${model}/"$3"
done

python -m tool.scripts.compute_metrics data/novels_titles/combined_set.txt data/testing_sets/"$1"/ experiments/"$2"/ner/nltk/ner_model_annotated/ experiments/"$2"/ner/nltk/"$3"
python -m tool.scripts.compute_metrics data/novels_titles/combined_set.txt data/testing_sets/"$1"/ experiments/"$2"/ner/stanza/ner_model_annotated/ experiments/"$2"/ner/stanza/"$3"

for model in ner ner-large ner-fast; do
  python -m tool.scripts.compute_metrics data/novels_titles/combined_set.txt data/testing_sets/"$1"/ experiments/"$2"/ner/flair__${model}/ner_model_annotated/ experiments/"$2"/ner/flair__${model}/"$3"
done
