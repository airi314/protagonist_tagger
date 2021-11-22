#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh

conda activate protagonist
for model in en_core_web_sm en_core_web_md en_core_web_lg; do
  python -m tool.scripts.compute_metrics data/novels_titles/combined_set.txt data/testing_sets/"$1"/ experiments/ner/spacy__${model}/ner_model_annotated/ experiments/ner/spacy__${model}/"$2"
done

python -m tool.scripts.compute_metrics data/novels_titles/combined_set.txt data/testing_sets/"$1"/ experiments/ner/nltk/ner_model_annotated/ experiments/ner/nltk/"$2"
python -m tool.scripts.compute_metrics data/novels_titles/combined_set.txt data/testing_sets/"$1"/ experiments/ner/stanza/ner_model_annotated/ experiments/ner/stanza/"$2"

for model in ner ner-large ner-fast; do
  python -m tool.scripts.compute_metrics data/novels_titles/combined_set.txt data/testing_sets/"$1"/ experiments/ner/flair__${model}/ner_model_annotated/ experiments/ner/flair__${model}/"$2"
done
