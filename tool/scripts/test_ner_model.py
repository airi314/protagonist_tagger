import argparse
import re
import os
import json

from tool.file_and_directory_management import read_file_to_list, read_sentences_from_file
from tool.data_generator import json_to_spacy_train_data, spacy_format_to_json
from tool.file_and_directory_management import dir_path, file_path
from tool.model.utils import load_model

# titles_path - path to .txt file with titles of novels from which the sampled data are to be generated (titles should
#       not contain any special characters and spaces should be replaced with "_", for example "Pride_andPrejudice")
# names_gold_standard_dir_path - path to directory with .txt files containing gold standard with annotations being full
#       names of literary characters (names of files should be the same as corresponding novels titles on the titles
#       list)
# generated_data_dir - directory where generated data should be stored
# testing_data_dir_path - directory containing .txt files with sentences extracted from novels to be included in the
#       testing process
# ner_model_dir_path - path to directory containing fine-tune NER model to be tested; if None standard spacy NER
#       model is used
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
    parser.add_argument('titles_path', type=file_path)
    parser.add_argument('testing_data_dir_path', type=dir_path)
    parser.add_argument('generated_data_dir', type=str)
    parser.add_argument('library', type=str, default='spacy', nargs='?')
    parser.add_argument('ner_model', type=str, default=None, nargs='?')
    opt = parser.parse_args()
    main(opt.titles_path, opt.testing_data_dir_path,
         opt.generated_data_dir, opt.library, opt.ner_model)
