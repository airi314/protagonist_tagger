#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh

conda activate protagonist
for subset in small large; do
	for model in en_core_web_sm en_core_web_md en_core_web_lg; do
		python -m tool.scripts.test_matcher_algorithm data/novels_titles/${subset}_set.txt data/lists_of_characters/${subset}_set data/testing_sets/test_${subset}/ experiments/protagonist/spacy__${model}/${subset}_set/ spacy ${model}
	done

	python -m tool.scripts.test_matcher_algorithm data/novels_titles/${subset}_set.txt data/lists_of_characters/${subset}_set data/testing_sets/test_${subset}/ experiments/protagonist/nltk/${subset}_set nltk
	python -m tool.scripts.test_matcher_algorithm data/novels_titles/${subset}_set.txt data/lists_of_characters/${subset}_set data/testing_sets/test_${subset}/ experiments/protagonist/stanza/${subset}_set stanza
done

conda activate protagonist2
for subset in small large; do
	for model in ner ner-large ner-fast; do
		python -m tool.scripts.test_matcher_algorithm data/novels_titles/${subset}_set.txt data/lists_of_characters/${subset}_set data/testing_sets/test_${subset}/ experiments/protagonist/flair__${model}/${subset}_set/ flair ${model}
	done
done