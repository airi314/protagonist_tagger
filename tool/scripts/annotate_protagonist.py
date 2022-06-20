import argparse
import os
import json
from tqdm import tqdm
import logging

from tool.names_matcher import NamesMatcher
from tool.preprocessing import get_test_data_for_novel, get_characters_for_novel
from tool.file_and_directory_management import write_text_to_file, read_file_to_list, dir_path, file_path, mkdir
from tool.annotations_utils import read_annotations


def main(titles_path, characters_lists_dir_path, testing_data_dir_path,
         generated_data_dir, library, ner_model, precision, matcher_rules,
         fix_personal_titles, gutenberg, pride_and_prejudice, save_ner):
    full_text = True if (gutenberg or pride_and_prejudice) else False
    titles = read_file_to_list(titles_path)
    names_matcher = NamesMatcher(
        precision,
        library,
        ner_model,
        matcher_rules,
        fix_personal_titles)
    mkdir(generated_data_dir)
    logging.basicConfig(
        filename=os.path.join(generated_data_dir, 'protagonist.log'),
        level=logging.DEBUG,
        filemode='w'
    )

    for title in tqdm(titles[9:]):
        names_matcher.logger.debug("TITLE: " + title)
        test_data = get_test_data_for_novel(
                title, testing_data_dir_path, gutenberg, pride_and_prejudice)
        characters = get_characters_for_novel(title, characters_lists_dir_path)
        ner_path = os.path.join(generated_data_dir, 'ner', title + ".json")

        if save_ner and os.path.exists(ner_path):
            names_matcher.logger.info("NER results loaded from: " + ner_path)
            ner_results = read_annotations(ner_path)
        else:
            names_matcher.logger.info("Getting NER results...")
            ner_results = names_matcher.model.get_ner_results(
                test_data, full_text)

        if save_ner and not os.path.exists(ner_path):
            write_text_to_file(ner_path, json.dumps(ner_results))

        matcher_results = names_matcher.recognize_person_entities(
            ner_results, characters)
        path = os.path.join(generated_data_dir, title + '.json')
        write_text_to_file(path, json.dumps(matcher_results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('titles_path', type=file_path,
                        help="path to .txt file with titles of novels")
    parser.add_argument('characters_lists_dir_path', type=dir_path,
                        help="path to the directory containing .txt files with novel characters")
    parser.add_argument('testing_data_dir_path', type=dir_path,
                        help="path to the directory containing .txt files with novel text")
    parser.add_argument('generated_data_dir', type=str,
                        help="directory where generated data should be stored")
    parser.add_argument('library', type=str, default='spacy', nargs='?',
                        help="library which should be used")
    parser.add_argument('ner_model', type=str, default=None, nargs='?',
                        help="model from chosen library which should be used")
    parser.add_argument('--precision', type=int, default=75,
                        help="precision threshold for string similarity")
    parser.add_argument('--fix_personal_titles', action='store_true',
                        help="boolean, if True then annotations will of personal titles will be fixed")
    parser.add_argument('--gutenberg', action='store_true',
                        help="boolean, if True then input file is preprocessed according to the Gutenberg ebooks file format")
    parser.add_argument('--pride_and_prejudice', action='store_true',
                        help="boolean, if True then input file is preprocessed according Pride_and_Prejudice corpus format")
    parser.add_argument('--save_ner', action='store_true',
                        help="boolean, if True then annotations of NER will be saved or loaded from disk if they exist")
    parser.add_argument('--not_check_diminutive', action='store_true',
                        help="boolean, if True check_diminutive rule is disabled")
    parser.add_argument('--not_match_personal_title_100', action='store_true',
                        help = "boolean, if True match_personal_title_100 rule is disabled")
    parser.add_argument('--not_match_personal_title', action='store_true',
                        help = "boolean, if True match_personal_title rule is disabled")
    parser.add_argument('--not_match_title_gender', action='store_true',
                        help="boolean, if True match_title_gender rule is disabled")
    parser.add_argument('--not_match_entity_gender', action='store_true',
                        help="boolean, if True match_entity_gender rule is disabled")

    opt = parser.parse_args()
    rules = {
        "check_diminutive": not opt.not_check_diminutive,
        "match_personal_title_100": not opt.not_match_personal_title_100,
        "match_personal_title": not opt.not_match_personal_title,
        "match_title_gender": not opt.not_match_title_gender,
        "match_entity_gender": not opt.not_match_entity_gender
    }

    main(opt.titles_path, opt.characters_lists_dir_path, opt.testing_data_dir_path,
         opt.generated_data_dir, opt.library, opt.ner_model, opt.precision,
         rules, opt.fix_personal_titles, opt.gutenberg,
         opt.pride_and_prejudice, opt.save_ner)
