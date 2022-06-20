import argparse
import os
from tqdm import tqdm
import json
import logging

from tool.file_and_directory_management import read_file_to_list, dir_path, \
    file_path, write_text_to_file, mkdir
from tool.coreference.utils import load_model
from tool.preprocessing import get_test_data_for_novel


def main(titles_path, testing_data_dir_path, generated_data_dir,
         library, model_name, save_singletons, gutenberg, save_conll,
         pride_and_prejudice):
    titles = read_file_to_list(titles_path)
    model = load_model(library, model_name, save_singletons)
    mkdir(generated_data_dir)
    logging.basicConfig(
        filename=os.path.join(generated_data_dir, 'coreference.log'),
        level=logging.DEBUG,
        filemode='w')

    for title in tqdm(titles[-1:]):
        test_data = get_test_data_for_novel(
                title, testing_data_dir_path, gutenberg, pride_and_prejudice)

        prefix = 0
        results = []
        for part in tqdm(test_data):
            if save_conll:
                lines_part, prefix = model.get_clusters(
                    part, prefix, True)
                results += lines_part
            else:
                results_part = model.get_clusters(part, prefix=prefix)
                results += results_part
                prefix += len(part) + 1

        if save_conll:
            path = os.path.join(generated_data_dir, title + ".conll")
            write_text_to_file(path, '\n'.join(results))
        else:
            results = {'content': '\n'.join(test_data), 'clusters': results}
            path = os.path.join(generated_data_dir, title + ".json")
            write_text_to_file(path, json.dumps(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('titles_path', type=file_path,
                        help="path to .txt file with titles of novels")
    parser.add_argument('testing_data_dir_path', type=dir_path,
                        help="path to the directory containing .txt files with novel text")
    parser.add_argument('generated_data_dir', type=str,
                        help="directory where generated data should be stored")
    parser.add_argument('library', type=str, default='coreferee', nargs='?',
                        help="library which should be used for CR")
    parser.add_argument('model_name', type=str, default=None, nargs='?',
                        help="model from chosen library which should be used")
    parser.add_argument('--save_singletons', action='store_true',
                        help="boolean; if True then clusters with single elements will be saved")
    parser.add_argument('--save_conll', action='store_true',
                        help='boolean; if True then results will be saved in conll format; by default there is json format')
    parser.add_argument('--gutenberg', action='store_true',
                        help="boolean, if True then input file is preprocessed according to the Gutenberg ebooks file format")
    parser.add_argument('--pride_and_prejudice', action='store_true',
                        help="boolean, if True then input file is preprocessed according Pride_and_Prejudice corpus format")

    opt = parser.parse_args()
    main(opt.titles_path, opt.testing_data_dir_path, opt.generated_data_dir,
         opt.library, opt.model_name, opt.save_singletons, opt.split_paragraphs,
         opt.gutenberg, opt.pride_and_prejudice)
