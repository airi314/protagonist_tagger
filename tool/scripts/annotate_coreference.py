import argparse
import os
from tqdm import tqdm

from tool.file_and_directory_management import read_file_to_list, dir_path, file_path
from tool.coreference.utils import load_model
from tool.preprocessing import get_litbank_text_parts


def main(titles_path, testing_data_dir_path, generated_data_dir,
         library, model_name):
    titles = read_file_to_list(titles_path)
    model = load_model(library, model_name)

    for title in tqdm(titles[:10]):

        text_parts = get_litbank_text_parts(
            os.path.join(testing_data_dir_path, title + '_brat.txt'))
        lines = []

        for i, text_part in tqdm(enumerate(text_parts)):
            lines += model.get_clusters(text_part)

        path = os.path.join(generated_data_dir, title + ".conll")
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path, 'w') as f:
            f.write('\n'.join(lines))


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
    opt = parser.parse_args()
    main(opt.titles_path, opt.testing_data_dir_path, opt.generated_data_dir,
         opt.library, opt.model_name)
