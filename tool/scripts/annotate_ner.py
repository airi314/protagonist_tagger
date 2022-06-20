import argparse
import os
import json
from tqdm import tqdm
import logging

from tool.file_and_directory_management import read_file_to_list, \
    dir_path, file_path, mkdir, write_text_to_file
from tool.model.utils import load_model
from tool.preprocessing import get_test_data_for_novel


def main(titles_path, testing_data_dir_path, generated_data_dir,
         library='spacy', ner_model=None, fix_personal_titles=False,
         gutenberg=False, pride_and_prejudice=False):
    full_text = True if (gutenberg or pride_and_prejudice) else False
    titles = read_file_to_list(titles_path)
    model = load_model(library, ner_model, False, fix_personal_titles)
    mkdir(generated_data_dir)
    logging.basicConfig(
        filename=os.path.join(
            generated_data_dir,
            'ner.log'),
        level=logging.DEBUG,
        filemode='w')

    for title in tqdm(titles):
        model.logger.debug("TITLE: " + title)
        test_data = get_test_data_for_novel(
            title, testing_data_dir_path, gutenberg, pride_and_prejudice)
        ner_result = model.get_ner_results(test_data, full_text)
        path = os.path.join(generated_data_dir, title + ".json")
        write_text_to_file(path, json.dumps(ner_result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('titles_path', type=file_path,
                        help="path to .txt file with titles of novels")
    parser.add_argument('testing_data_dir_path', type=dir_path,
                        help="path to the directory containing .txt files with novel text")
    parser.add_argument('generated_data_dir', type=str,
                        help="directory where generated data should be stored")
    parser.add_argument('library', type=str, default='spacy', nargs='?',
                        help="library which should be used")
    parser.add_argument('ner_model', type=str, default=None, nargs='?',
                        help="model from chosen library which should be used")
    parser.add_argument('--fix_personal_titles', action='store_true',
                        help="boolean, if True then annotations will of personal titles will be fixed")
    parser.add_argument('--gutenberg', action='store_true',
                        help="boolean, if True then input file is preprocessed according to the Gutenberg ebooks file format")
    parser.add_argument('--pride_and_prejudice', action='store_true',
                        help="boolean, if True then input file is preprocessed according Pride_and_Prejudice corpus format")
    opt = parser.parse_args()
    main(opt.titles_path, opt.testing_data_dir_path,
         opt.generated_data_dir, opt.library, opt.ner_model,
         opt.fix_personal_titles,  opt.gutenberg, opt.pride_and_prejudice)
