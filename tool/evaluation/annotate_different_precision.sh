#!/bin/bash
set -e

for i in {1..100}; do

python -m tool.scripts.annotate_protagonist data/novels_titles/combined_set.txt data/lists_of_characters data/testing_sets/test/ experiments/rules/ flair ner-large --fix_personal_titles --save_ner --not_match_personal_title --precision $i
python -m tool.scripts.compute_metrics data/novels_titles/combined_set.txt data/testing_sets/test_names_gold_standard_corrected/ experiments/rules/ experiments/rules/stats_precision_${i} --protagonist_tagger

done