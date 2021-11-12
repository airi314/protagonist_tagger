#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh

conda activate protagonist
for subset in small large; do
	for model in en_core_web_sm en_core_web_md en_core_web_lg; do
		python -m tool.scripts.compute_metrics data/novels_titles/${subset}_set.txt data/testing_sets/test_${subset}_person_gold_standard/ experiments/ner/spacy__${model}/${subset}_set/ner_model_annotated/ experiments/ner/spacy__${model}/${subset}_set/stats
	done

	python -m tool.scripts.compute_metrics data/novels_titles/${subset}_set.txt data/testing_sets/test_${subset}_person_gold_standard/ experiments/ner/nltk/${subset}_set/ner_model_annotated/ experiments/ner/nltk/${subset}_set/stats
	python -m tool.scripts.compute_metrics data/novels_titles/${subset}_set.txt data/testing_sets/test_${subset}_person_gold_standard/ experiments/ner/stanza/${subset}_set/ner_model_annotated/ experiments/ner/stanza/${subset}_set/stats

	for model in ner ner-large ner-fast; do
		python -m tool.scripts.compute_metrics data/novels_titles/${subset}_set.txt data/testing_sets/test_${subset}_person_gold_standard/ experiments/ner/flair__${model}/${subset}_set/ner_model_annotated/ experiments/ner/flair__${model}/${subset}_set/stats
	done
done