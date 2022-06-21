#!/bin/bash
set -e

echo stats_1
python -m tool.scripts.annotate_protagonist data/novels_titles/combined_set.txt data/lists_of_characters data/testing_sets/test/ experiments/rules/ flair ner-large  --fix_personal_titles --save_ner --not_match_personal_title_100  --not_match_personal_title --not_match_title_gender --not_match_entity_gender --not_check_diminutive
python -m tool.scripts.compute_metrics data/novels_titles/combined_set.txt data/testing_sets/test_names_gold_standard_corrected/ experiments/rules/ experiments/rules/stats_1 --protagonist_tagger --print_results

echo stats_2
python -m tool.scripts.annotate_protagonist data/novels_titles/combined_set.txt data/lists_of_characters data/testing_sets/test/ experiments/rules/ flair ner-large  --fix_personal_titles --save_ner  --not_match_personal_title --not_match_title_gender --not_match_entity_gender --not_check_diminutive
python -m tool.scripts.compute_metrics data/novels_titles/combined_set.txt data/testing_sets/test_names_gold_standard_corrected/ experiments/rules/ experiments/rules/stats_2 --protagonist_tagger --print_results

echo stats_3
python -m tool.scripts.annotate_protagonist data/novels_titles/combined_set.txt data/lists_of_characters data/testing_sets/test/ experiments/rules/ flair ner-large  --fix_personal_titles --save_ner --not_match_personal_title_100  --not_match_title_gender --not_match_entity_gender --not_check_diminutive
python -m tool.scripts.compute_metrics data/novels_titles/combined_set.txt data/testing_sets/test_names_gold_standard_corrected/ experiments/rules/ experiments/rules/stats_3 --protagonist_tagger --print_results

echo stats_4
python -m tool.scripts.annotate_protagonist data/novels_titles/combined_set.txt data/lists_of_characters data/testing_sets/test/ experiments/rules/ flair ner-large  --fix_personal_titles --save_ner --not_match_personal_title_100  --not_match_personal_title --not_match_entity_gender --not_check_diminutive
python -m tool.scripts.compute_metrics data/novels_titles/combined_set.txt data/testing_sets/test_names_gold_standard_corrected/ experiments/rules/ experiments/rules/stats_4 --protagonist_tagger --print_results

echo stats_5
python -m tool.scripts.annotate_protagonist data/novels_titles/combined_set.txt data/lists_of_characters data/testing_sets/test/ experiments/rules/ flair ner-large  --fix_personal_titles --save_ner --not_match_personal_title_100  --not_match_personal_title --not_match_title_gender --not_check_diminutive
python -m tool.scripts.compute_metrics data/novels_titles/combined_set.txt data/testing_sets/test_names_gold_standard_corrected/ experiments/rules/ experiments/rules/stats_5 --protagonist_tagger --print_results

echo stats_6
python -m tool.scripts.annotate_protagonist data/novels_titles/combined_set.txt data/lists_of_characters data/testing_sets/test/ experiments/rules/ flair ner-large  --fix_personal_titles --save_ner --not_match_personal_title_100  --not_match_personal_title --not_match_title_gender --not_match_entity_gender
python -m tool.scripts.compute_metrics data/novels_titles/combined_set.txt data/testing_sets/test_names_gold_standard_corrected/ experiments/rules/ experiments/rules/stats_6 --protagonist_tagger --print_results

echo stats_7
python -m tool.scripts.annotate_protagonist data/novels_titles/combined_set.txt data/lists_of_characters data/testing_sets/test/ experiments/rules/ flair ner-large  --fix_personal_titles --save_ner --not_match_personal_title --not_match_title_gender --not_match_entity_gender
python -m tool.scripts.compute_metrics data/novels_titles/combined_set.txt data/testing_sets/test_names_gold_standard_corrected/ experiments/rules/ experiments/rules/stats_7 --protagonist_tagger --print_results

echo stats_8
python -m tool.scripts.annotate_protagonist data/novels_titles/combined_set.txt data/lists_of_characters data/testing_sets/test/ experiments/rules/ flair ner-large  --fix_personal_titles --save_ner --not_match_personal_title_100 --not_match_title_gender --not_match_entity_gender
python -m tool.scripts.compute_metrics data/novels_titles/combined_set.txt data/testing_sets/test_names_gold_standard_corrected/ experiments/rules/ experiments/rules/stats_8 --protagonist_tagger --print_results

echo stats_9
python -m tool.scripts.annotate_protagonist data/novels_titles/combined_set.txt data/lists_of_characters data/testing_sets/test/ experiments/rules/ flair ner-large  --fix_personal_titles --save_ner --not_match_personal_title_100  --not_match_personal_title --not_match_entity_gender
python -m tool.scripts.compute_metrics data/novels_titles/combined_set.txt data/testing_sets/test_names_gold_standard_corrected/ experiments/rules/ experiments/rules/stats_9 --protagonist_tagger --print_results

echo stats_10
python -m tool.scripts.annotate_protagonist data/novels_titles/combined_set.txt data/lists_of_characters data/testing_sets/test/ experiments/rules/ flair ner-large  --fix_personal_titles --save_ner --not_match_personal_title_100  --not_match_personal_title --not_match_title_gender
python -m tool.scripts.compute_metrics data/novels_titles/combined_set.txt data/testing_sets/test_names_gold_standard_corrected/ experiments/rules/ experiments/rules/stats_10 --protagonist_tagger --print_results

echo stats_11
python -m tool.scripts.annotate_protagonist data/novels_titles/combined_set.txt data/lists_of_characters data/testing_sets/test/ experiments/rules/ flair ner-large  --fix_personal_titles --save_ner --not_match_title_gender --not_match_entity_gender
python -m tool.scripts.compute_metrics data/novels_titles/combined_set.txt data/testing_sets/test_names_gold_standard_corrected/ experiments/rules/ experiments/rules/stats_11 --protagonist_tagger --print_results

echo stats_12
python -m tool.scripts.annotate_protagonist data/novels_titles/combined_set.txt data/lists_of_characters data/testing_sets/test/ experiments/rules/ flair ner-large  --fix_personal_titles --save_ner --not_match_personal_title --not_match_entity_gender
python -m tool.scripts.compute_metrics data/novels_titles/combined_set.txt data/testing_sets/test_names_gold_standard_corrected/ experiments/rules/ experiments/rules/stats_12 --protagonist_tagger --print_results

echo stats_13
python -m tool.scripts.annotate_protagonist data/novels_titles/combined_set.txt data/lists_of_characters data/testing_sets/test/ experiments/rules/ flair ner-large  --fix_personal_titles --save_ner --not_match_personal_title --not_match_title_gender
python -m tool.scripts.compute_metrics data/novels_titles/combined_set.txt data/testing_sets/test_names_gold_standard_corrected/ experiments/rules/ experiments/rules/stats_13 --protagonist_tagger --print_results

echo stats_14
python -m tool.scripts.annotate_protagonist data/novels_titles/combined_set.txt data/lists_of_characters data/testing_sets/test/ experiments/rules/ flair ner-large  --fix_personal_titles --save_ner --not_match_entity_gender
python -m tool.scripts.compute_metrics data/novels_titles/combined_set.txt data/testing_sets/test_names_gold_standard_corrected/ experiments/rules/ experiments/rules/stats_14 --protagonist_tagger --print_results

echo stats_15
python -m tool.scripts.annotate_protagonist data/novels_titles/combined_set.txt data/lists_of_characters data/testing_sets/test/ experiments/rules/ flair ner-large  --fix_personal_titles --save_ner --not_match_personal_title
python -m tool.scripts.compute_metrics data/novels_titles/combined_set.txt data/testing_sets/test_names_gold_standard_corrected/ experiments/rules/ experiments/rules/stats_15 --protagonist_tagger --print_results

echo stats_16
python -m tool.scripts.annotate_protagonist data/novels_titles/combined_set.txt data/lists_of_characters data/testing_sets/test/ experiments/rules/ flair ner-large  --fix_personal_titles --save_ner
python -m tool.scripts.compute_metrics data/novels_titles/combined_set.txt data/testing_sets/test_names_gold_standard_corrected/ experiments/rules/ experiments/rules/stats_16 --protagonist_tagger --print_results
