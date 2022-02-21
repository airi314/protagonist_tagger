#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh

conda activate protagonist
for model in en_core_web_sm en_core_web_md en_core_web_lg; do
  python -m tool.scripts.compute_metrics data/novels_titles/combined_set.txt data/testing_sets/"$1"/ experiments/"$2"/spacy__${model}/ner_model_annotated/ experiments/"$2"/spacy__${model}/"$3" --protagonist_tagger
done

python -m tool.scripts.compute_metrics data/novels_titles/combined_set.txt data/testing_sets/"$1"/ experiments/"$2"/nltk/ner_model_annotated/ experiments/"$2"/nltk/"$3" --protagonist_tagger
python -m tool.scripts.compute_metrics data/novels_titles/combined_set.txt data/testing_sets/"$1"/ experiments/"$2"/stanza/ner_model_annotated/ experiments/"$2"/stanza/"$3" --protagonist_tagger

for model in ner ner-large ner-fast; do
  python -m tool.scripts.compute_metrics data/novels_titles/combined_set.txt data/testing_sets/"$1"/ experiments/"$2"/flair__${model}/ner_model_annotated/ experiments/"$2"/flair__${model}/"$3" --protagonist_tagger
done
