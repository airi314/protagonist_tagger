import argparse
import os
from tqdm import tqdm
import json
import logging

from tool.file_and_directory_management import read_file_to_list, dir_path, file_path, write_text_to_file, mkdir
from tool.coreference.utils import load_model
from tool.preprocessing import get_litbank_parts, get_litbank_text, get_pride_and_prejudice


def main(titles_path, testing_data_dir_path, generated_data_dir,
         library, model_name, save_singletons, split_paragraphs, save_conll,
         pride_and_prejudice):
    titles = read_file_to_list(titles_path)
    model = load_model(library, model_name, save_singletons)
    logging.basicConfig(
        filename=os.path.join(generated_data_dir, 'coreference.log'),
        level=logging.DEBUG,
        filemode='w')

    for title in tqdm(titles):
        path = os.path.join(testing_data_dir_path, title + '.txt')
        if pride_and_prejudice:
            text_parts = [
                get_pride_and_prejudice(
                    title,
                    testing_data_dir_path,
                    True)]
        elif split_paragraphs:
            text_parts = get_litbank_parts(path)
        else:
            text_parts = [get_litbank_text(path)]

        start_cluster_id = 0
        results = []
        for part in tqdm(text_parts):
            if save_conll:
                lines_part, start_cluster_id = model.get_clusters(
                    part, start_cluster_id, True)
                results += lines_part
            else:
                results_part = model.get_clusters(part)
                results.append(results_part)

        mkdir(generated_data_dir)

        if save_conll:
            path = os.path.join(generated_data_dir, title + ".conll")
            write_text_to_file(path, '\n'.join(results))
        else:
            path = os.path.join(generated_data_dir, title + ".json")
            write_text_to_file(path, json.dumps(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('titles_path', type=file_path,
                        help="path to .txt file with titles of novels for which NER model should be tested")
    parser.add_argument('testing_data_dir_path', type=dir_path,
                        help="path to the directory containing .txt files with sentences extracted from novels "
                             "to be included in the testing process")
    parser.add_argument('generated_data_dir', type=str,
                        help="directory where generated data should be stored")
    parser.add_argument('library', type=str, default='coreferee', nargs='?',
                        help="library which should be used for coreferences")
    parser.add_argument('model_name', type=str, default=None, nargs='?',
                        help="model from chosen library which should be used")
    parser.add_argument('--save_singletons', action='store_true')
    parser.add_argument('--split_paragraphs', action='store_true')
    parser.add_argument('--save_conll', action='store_true')
    parser.add_argument('--pride_and_prejudice', action='store_true')

    opt = parser.parse_args()
    main(opt.titles_path, opt.testing_data_dir_path, opt.generated_data_dir,
         opt.library, opt.model_name, opt.save_singletons, opt.split_paragraphs,
         opt.save_conll, opt.pride_and_prejudice)
