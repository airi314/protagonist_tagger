import argparse
import os
import json
from tqdm import tqdm
import logging

from tool.names_matcher import NamesMatcher
from tool.preprocessing import get_test_data_for_novel, get_characters_for_novel
from tool.file_and_directory_management import open_path, read_file_to_list, dir_path, file_path, mkdir
from tool.annotations_utils import read_annotations


def main(titles_path, characters_lists_dir_path, testing_data_dir_path,
         generated_data_dir, library, ner_model, precision, matcher_rules,
         fix_personal_titles, full_text, save_ner):
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

    for title in tqdm(titles):
        names_matcher.logger.debug("TITLE: " + title)
        test_data = get_test_data_for_novel(
            title, testing_data_dir_path, full_text)
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
            open_path(ner_path, 'w').write(json.dumps(ner_results))

        matcher_results = names_matcher.recognize_person_entities(
            ner_results, characters)
        path = os.path.join(generated_data_dir, title + '.json')
        open_path(path, 'w').write(json.dumps(matcher_results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('titles_path', type=file_path,
                        help="path to .txt file with titles of novels for which NER model should be tested")
    parser.add_argument('characters_lists_dir_path', type=dir_path,
                        help="path to the directory containing .txt files with novel characters")
    parser.add_argument('testing_data_dir_path', type=dir_path,
                        help="path to the directory containing .txt files with sentences extracted from novels "
                             "to be included in the testing process")
    parser.add_argument('generated_data_dir', type=str,
                        help="directory where generated data should be stored")
    parser.add_argument('library', type=str, default='spacy', nargs='?',
                        help="library which should be used to test NER")
    parser.add_argument('ner_model', type=str, default=None, nargs='?',
                        help="model from chosen library which should be used to test NER")
    parser.add_argument('--precision', type=int, default=75,
                        help="precision threshold for string similarity")
    parser.add_argument('--fix_personal_titles', action='store_true')
    parser.add_argument('--full_text', action='store_true')
    parser.add_argument('--save_ner', action='store_true')
    parser.add_argument('--not_check_diminutive', action='store_true')
    parser.add_argument('--not_match_personal_title_100', action='store_true')
    parser.add_argument('--not_match_personal_title', action='store_true')
    parser.add_argument('--not_match_title_gender', action='store_true')
    parser.add_argument('--not_match_entity_gender', action='store_true')

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
         rules, opt.fix_personal_titles, opt.full_text, opt.save_ner)
