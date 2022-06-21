#!/bin/bash
set -e

for model in fast_coref long_doc_coref wl_coref; do
  python -m tool.scripts.annotate_coreference data/novels_titles/litbank_test.txt  data/testing_sets/test_litbank experiments/coreference/${model} ${model} --save_conll
done;

python -m tool.scripts.annotate_coreference data/novels_titles/litbank_test.txt  data/testing_sets/test_litbank experiments/coreference/e2e_coref__d1_1 e2e_coref train_spanbert_large_ml0_d1/May08_12-37-39_54000 --save_conll --split_paragraphs
python -m tool.scripts.annotate_coreference data/novels_titles/litbank_test.txt  data/testing_sets/test_litbank experiments/coreference/e2e_coref__d1_2 e2e_coref train_spanbert_large_ml0_d1/May10_03-28-49_54000 --save_conll --split_paragraphs
python -m tool.scripts.annotate_coreference data/novels_titles/litbank_test.txt  data/testing_sets/test_litbank experiments/coreference/e2e_coref__d2_1 e2e_coref train_spanbert_large_ml0_d2/May08_12-24-06_64000 --save_conll --split_paragraphs
python -m tool.scripts.annotate_coreference data/novels_titles/litbank_test.txt  data/testing_sets/test_litbank experiments/coreference/e2e_coref__d2_2 e2e_coref train_spanbert_large_ml0_d2/May08_12-38-29_58000 --save_conll --split_paragraphs
python -m tool.scripts.annotate_coreference data/novels_titles/litbank_test.txt  data/testing_sets/test_litbank experiments/coreference/e2e_coref__sc_1 e2e_coref train_spanbert_large_ml0_sc/May08_12-24-28_44000 --save_conll --split_paragraphs
python -m tool.scripts.annotate_coreference data/novels_titles/litbank_test.txt  data/testing_sets/test_litbank experiments/coreference/e2e_coref__sc_2 e2e_coref train_spanbert_large_ml0_sc/May08_12-38-54_57000 --save_conll --split_paragraphs
python -m tool.scripts.annotate_coreference data/novels_titles/litbank_test.txt  data/testing_sets/test_litbank experiments/coreference/e2e_coref__cm_1 e2e_coref train_spanbert_large_ml0_cm_fn1000_max_dloss/May14_05-15-38_63000 --save_conll --split_paragraphs
python -m tool.scripts.annotate_coreference data/novels_titles/litbank_test.txt  data/testing_sets/test_litbank experiments/coreference/e2e_coref__cm_2 e2e_coref train_spanbert_large_ml0_cm_fn1000_max_dloss/May22_23-31-16_66000 --save_conll --split_paragraphs


for model in fast_coref long_doc_coref wl_coref; do
  python -m tool.scripts.compute_metrics data/novels_titles/litbank_test.txt  data/testing_sets/test_litbank experiments/coreference/${model} experiments/coreference/${model}/stats --print_results --coreference_resolution
done;

python -m tool.scripts.compute_metrics data/novels_titles/litbank_test.txt  data/testing_sets/test_litbank experiments/coreference/e2e_coref__d1_1 experiments/coreference/e2e_coref__d1_1/stats --print_results --coreference_resolution
python -m tool.scripts.compute_metrics data/novels_titles/litbank_test.txt  data/testing_sets/test_litbank experiments/coreference/e2e_coref__d1_2 experiments/coreference/e2e_coref__d1_2/stats --print_results --coreference_resolution
python -m tool.scripts.compute_metrics data/novels_titles/litbank_test.txt  data/testing_sets/test_litbank experiments/coreference/e2e_coref__d2_1 experiments/coreference/e2e_coref__d2_1/stats --print_results --coreference_resolution
python -m tool.scripts.compute_metrics data/novels_titles/litbank_test.txt  data/testing_sets/test_litbank experiments/coreference/e2e_coref__d2_2 experiments/coreference/e2e_coref__d2_2/stats --print_results --coreference_resolution
python -m tool.scripts.compute_metrics data/novels_titles/litbank_test.txt  data/testing_sets/test_litbank experiments/coreference/e2e_coref__sc_1 experiments/coreference/e2e_coref__sc_1/stats --print_results --coreference_resolution
python -m tool.scripts.compute_metrics data/novels_titles/litbank_test.txt  data/testing_sets/test_litbank experiments/coreference/e2e_coref__sc_2 experiments/coreference/e2e_coref__sc_2/stats --print_results --coreference_resolution
python -m tool.scripts.compute_metrics data/novels_titles/litbank_test.txt  data/testing_sets/test_litbank experiments/coreference/e2e_coref__cm_1 experiments/coreference/e2e_coref__cm_1/stats --print_results --coreference_resolution
python -m tool.scripts.compute_metrics data/novels_titles/litbank_test.txt  data/testing_sets/test_litbank experiments/coreference/e2e_coref__cm_2 experiments/coreference/e2e_coref__cm_2/stats --print_results --coreference_resolution

