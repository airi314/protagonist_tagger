#!/bin/bash
set -e

# argument1 - directory name with goldstandard person annotation inside 'data/testing_sets' directory
# argument2 - name of experiment; subdirectory of 'experiments' directory
# argument3 - name of stats directory

for model in en_core_web_sm en_core_web_md en_core_web_lg en_core_web_trf; do
  python -m tool.scripts.compute_metrics data/novels_titles/combined_set.txt data/testing_sets/"$1"/ experiments/"$2"/spacy__${model}/ experiments/"$2"/spacy__${model}/"$3"
done

python -m tool.scripts.compute_metrics data/novels_titles/combined_set.txt data/testing_sets/"$1"/ experiments/"$2"/nltk/ experiments/"$2"/nltk/"$3"

for model in ontonotes conll03; do
	python -m tool.scripts.compute_metrics data/novels_titles/combined_set.txt data/testing_sets/"$1"/ experiments/"$2"/stanza__${model}/ experiments/"$2"/stanza__${model}/"$3"
done

for model in ner ner-large ner-fast; do
  python -m tool.scripts.compute_metrics data/novels_titles/combined_set.txt data/testing_sets/"$1"/ experiments/"$2"/flair__${model}/ experiments/"$2"/flair__${model}/"$3"
done
