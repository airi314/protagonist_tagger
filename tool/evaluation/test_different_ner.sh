#!/bin/bash
conda activate protagonist2
for subset in small large; do
	for model in ner ner-large ner-fast; do
		python -m tool.scripts.test_ner_model data/novels_titles/${subset}_set.txt data/testing_sets/test_${subset}_person_gold_standard/ data/testing_sets/test_${subset}/ experiments/ner/flair-${model}/${subset}_set/ flair ${model}
	done
done 

conda activate protagonist
for subset in small large; do
	for model in en_core_web_sm en_core_web_md en_core_web_lg; do
		python -m tool.scripts.test_ner_model data/novels_titles/${subset}_set.txt data/testing_sets/test_${subset}_person_gold_standard/ data/testing_sets/test_${subset}/ experiments/ner/spacy-${model}/${subset}_set/ spacy ${model}
	done

	python -m tool.scripts.test_ner_model data/novels_titles/${subset}_set.txt data/testing_sets/test_${subset}_person_gold_standard/ data/testing_sets/test_${subset}/ experiments/ner/nltk/${subset}_set nltk None
	python -m tool.scripts.test_ner_model data/novels_titles/${subset}_set.txt data/testing_sets/test_${subset}_person_gold_standard/ data/testing_sets/test_${subset}/ experiments/ner/stanza/${subset}_set stanza None
done