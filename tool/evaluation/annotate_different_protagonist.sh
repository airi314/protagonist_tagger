#!/bin/bash
set -e

# argument1 - name of experiment; subdirectory of 'experiments' directory
# argument2 - flag; if '--fix_personal_titles' then personal titles annotations will be fixed

for model in en_core_web_sm en_core_web_md en_core_web_lg en_core_web_trf; do
  python -m tool.scripts.annotate_protagonist data/novels_titles/combined_set.txt data/lists_of_characters data/testing_sets/test/ experiments/"$1"/spacy__${model}/ spacy ${model} "${@:2}"
done

python -m tool.scripts.annotate_protagonist data/novels_titles/combined_set.txt data/lists_of_characters data/testing_sets/test/ experiments/"$1"/nltk/ nltk "${@:2}"

for model in ontonotes conll03; do
	python -m tool.scripts.annotate_ner data/novels_titles/combined_set.txt data/lists_of_characters data/testing_sets/test experiments/"$1"/stanza__${model}/ stanza ${model} "${@:2}"
done

for model in ner ner-large ner-fast; do
  python -m tool.scripts.annotate_protagonist data/novels_titles/combined_set.txt data/lists_of_characters data/testing_sets/test/ experiments/"$1"/flair__${model}/ flair ${model} "${@:2}"
done
