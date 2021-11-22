#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh

conda activate protagonist
for model in en_core_web_sm en_core_web_md en_core_web_lg; do
  python -m tool.scripts.compute_metrics data/novels_titles/combined_set.txt data/testing_sets/"$1"/ experiments/protagonist/spacy__${model}/ner_model_annotated/ experiments/protagonist/spacy__${model}/"$2" --protagonist_tagger
done

python -m tool.scripts.compute_metrics data/novels_titles/combined_set.txt data/testing_sets/"$1"/ experiments/protagonist/nltk/ner_model_annotated/ experiments/protagonist/nltk/"$2" --protagonist_tagger
python -m tool.scripts.compute_metrics data/novels_titles/combined_set.txt data/testing_sets/"$1"/ experiments/protagonist/stanza/ner_model_annotated/ experiments/protagonist/stanza/"$2" --protagonist_tagger

for model in ner ner-large ner-fast; do
  python -m tool.scripts.compute_metrics data/novels_titles/combined_set.txt data/testing_sets/"$1"/ experiments/protagonist/flair__${model}/ner_model_annotated/ experiments/protagonist/flair__${model}/"$2" --protagonist_tagger
done
