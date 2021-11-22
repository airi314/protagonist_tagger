#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh

conda activate protagonist
for model in en_core_web_sm en_core_web_md en_core_web_lg; do
  python -m tool.scripts.test_matcher_algorithm data/novels_titles/combined_set.txt data/lists_of_characters data/testing_sets/test/ experiments/protagonist/spacy__${model}/ spacy ${model}
done

python -m tool.scripts.test_matcher_algorithm data/novels_titles/combined_set.txt data/lists_of_characters data/testing_sets/test/ experiments/protagonist/nltk/ nltk
python -m tool.scripts.test_matcher_algorithm data/novels_titles/combined_set.txt data/lists_of_characters data/testing_sets/test/ experiments/protagonist/stanza/ stanza

conda activate protagonist2
for model in ner ner-large ner-fast; do
  python -m tool.scripts.test_matcher_algorithm data/novels_titles/combined_set.txt data/lists_of_characters data/testing_sets/test/ experiments/protagonist/flair__${model}/ flair ${model}
done
