import argparse
import os
import json

from tool.file_and_directory_management import read_file_to_list, read_sentences_from_file, dir_path, file_path
from tool.model.utils import load_model


def main(titles_path, testing_data_dir_path, generated_data_dir, library='spacy', ner_model=None):
    titles = read_file_to_list(titles_path)

    model = load_model(library, ner_model, False)

    for title in titles:
        test_data = read_sentences_from_file(os.path.join(testing_data_dir_path, title))
        ner_result = model.get_ner_results(test_data)

        path = os.path.join(generated_data_dir, "ner_model_annotated", title + ".json")

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        with open(path, 'w+') as result:
            json.dump(ner_result, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('titles_path', type=file_path,
                        description="path to .txt file with titles of novels for which NER model should be tested")
    parser.add_argument('testing_data_dir_path', type=dir_path,
                        description="path to the directory containing .txt files with sentences extracted from novels "
                                    "to be included in the testing process")
    parser.add_argument('generated_data_dir', type=str,
                        description="directory where generated data should be stored")
    parser.add_argument('library', type=str, default='spacy', nargs='?',
                        description="library which should be used to test NER")
    parser.add_argument('ner_model', type=str, default=None, nargs='?',
                        description="model from chosen library which should be used to test NER")
    opt = parser.parse_args()
    main(opt.titles_path, opt.testing_data_dir_path,
         opt.generated_data_dir, opt.library, opt.ner_model)
