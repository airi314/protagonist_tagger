import argparse

from tool.ner_metrics import characters_tags_metrics, ner_metrics
from tool.file_and_directory_management import dir_path, file_path


# titles_path - path to .txt file with titles of novels which should be included in the analysis (titles should
#       not contain any special characters and spaces should be replaced with "_", for example "Pride_andPrejudice")
# gold_standard_dir_path - path to directory containing gold standard for the testing set (names of files should be
#       the same as titles on the list)
# testing_set_dir_path - path to directory containing testing sets (selected sentences) extracted from novels, each in
#       file with the same name as corresponding title of the novel
# stats_dir - directory where the computed metrics should be stored
# protagonist_tagger - if true metrics are calculated for sets annotated with names of literary characters (computing
#       metrics for protagonistTagger performance); if false metrics are calculated for general tag person (computing
#       metrics for NER model performance)
def main(titles_path, gold_standard_dir_path, testing_set_dir_path, stats_dir, protagonist_tagger=True):
    if protagonist_tagger:
        characters_tags_metrics(titles_path, gold_standard_dir_path, testing_set_dir_path, stats_dir)
    else:
        ner_metrics(titles_path, gold_standard_dir_path, testing_set_dir_path, stats_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('titles_path', type=file_path)
    parser.add_argument('gold_standard_dir_path', type=dir_path)
    parser.add_argument('testing_set_dir_path', type=dir_path)
    parser.add_argument('stats_dir', type=str)
    parser.add_argument('--protagonist_tagger', action='store_true')
    opt = parser.parse_args()
    main(opt.titles_path, opt.gold_standard_dir_path, opt.testing_set_dir_path, opt.stats_dir, opt.protagonist_tagger)
