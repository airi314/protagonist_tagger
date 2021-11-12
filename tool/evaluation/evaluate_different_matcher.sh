#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh

conda activate protagonist
for subset in small large; do
	for model in en_core_web_sm en_core_web_md en_core_web_lg; do
		python -m tool.scripts.compute_metrics data/novels_titles/${subset}_set.txt data/testing_sets/test_${subset}_names_gold_standard/ experiments/protagonist/spacy__${model}/${subset}_set/ experiments/protagonist/spacy__${model}/${subset}_set/stats --protagonist_tagger
	done

	python -m tool.scripts.compute_metrics data/novels_titles/${subset}_set.txt data/testing_sets/test_${subset}_names_gold_standard/ experiments/protagonist/nltk/${subset}_set/ experiments/protagonist/nltk/${subset}_set/stats --protagonist_tagger
	python -m tool.scripts.compute_metrics data/novels_titles/${subset}_set.txt data/testing_sets/test_${subset}_names_gold_standard/ experiments/protagonist/stanza/${subset}_set/ experiments/protagonist/stanza/${subset}_set/stats --protagonist_tagger

	for model in ner ner-large ner-fast; do
		python -m tool.scripts.compute_metrics data/novels_titles/${subset}_set.txt data/testing_sets/test_${subset}_names_gold_standard/ experiments/protagonist/flair__${model}/${subset}_set/ experiments/protagonist/flair__${model}/${subset}_set/stats --protagonist_tagger
	done
done