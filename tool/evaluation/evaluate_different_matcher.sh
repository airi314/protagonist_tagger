#!/bin/bash
conda activate protagonist
for subset in small large; do
	for model in ner ner-large ner-fast; do
		python -m tool.scripts.compute_metrics data/novels_titles/${subset}_set.txt data/testing_sets/test_${subset}_names_gold_standard/ experiments/protagonist/flair-${model}/${subset}_set/ experiments/protagonist/flair-${model}/${subset}_set/stats True
	done

	for model in en_core_web_sm en_core_web_md en_core_web_lg; do
		python -m tool.scripts.compute_metrics data/novels_titles/${subset}_set.txt data/testing_sets/test_${subset}_names_gold_standard/ experiments/protagonist/spacy-${model}/${subset}_set/ experiments/protagonist/spacy-${model}/${subset}_set/stats True
	done

	python -m tool.scripts.compute_metrics data/novels_titles/${subset}_set.txt data/testing_sets/test_${subset}_names_gold_standard/ experiments/protagonist/nltk/${subset}_set/ experiments/protagonist/nltk/${subset}_set/stats True
	python -m tool.scripts.compute_metrics data/novels_titles/${subset}_set.txt data/testing_sets/test_${subset}_names_gold_standard/ experiments/protagonist/stanza/${subset}_set/ experiments/protagonist/stanza/${subset}_set/stats True
done