import sys
import re
import os
import json
from tool.file_and_directory_management import read_file_to_list, read_sentences_from_file
from tool.data_generator import json_to_spacy_train_data, spacy_format_to_json


def load_model(library, ner_model_dir_path):
    if library == 'spacy':
        from tool.model.spacy_model import SpacyModel
        if ner_model_dir_path is None:
            ner_model_dir_path = 'en_core_web_sm'
        model = SpacyModel(ner_model_dir_path)

    elif library == 'nltk':
        from tool.model.nltk_model import NLTKModel
        model = NLTKModel()

    elif library == 'stanza':
        from tool.model.stanza_model import StanzaModel
        model = StanzaModel()

    elif library == 'flair':
        from tool.model.flair_model import FlairModel
        if ner_model_dir_path is None:
            ner_model_dir_path = 'flair'
        model = FlairModel(ner_model_dir_path)

    return model


def generalize_tags(data):
    return re.sub(
        r"([0-9]+,\s[0-9]+,\s')[a-zA-Z\s\.àâäèéêëîïôœùûüÿçÀÂÄÈÉÊËÎÏÔŒÙÛÜŸÇ]+", r"\1PERSON", str(data))


# generalizing annotations - changing tags containing full names of literary characters to general tag PERSON
# titles - list of novels titles to be inlcluded in the generated data set (titles should not contain any special
#       characters and spaces should be replaced with "_", for example "Pride_andPrejudice")
# names_gold_standard_dir_path - path to directory with .txt files containing gold standard with annotations being full
#       names of literary characters (names of files should be the same as corresponding novels titles on the titles
#       list)
# generated_data_dir - directory where generated data should be stored
def generate_generalized_data(
        titles, names_gold_standard_dir_path, generated_data_dir):
    for title in titles:
        test_data = json_to_spacy_train_data(os.path.join(
            names_gold_standard_dir_path, title + ".json"))
        generalized_test_data = generalize_tags(test_data)
        spacy_format_to_json(
            os.path.join(
                generated_data_dir,
                "generated_gold_standard"),
            generalized_test_data,
            title)


# titles_path - path to .txt file with titles of novels from which the sampled data are to be generated (titles should
#       not contain any special characters and spaces should be replaced with "_", for example "Pride_andPrejudice")
# names_gold_standard_dir_path - path to directory with .txt files containing gold standard with annotations being full
#       names of literary characters (names of files should be the same as corresponding novels titles on the titles
#       list)
# generated_data_dir - directory where generated data should be stored
# testing_data_dir_path - directory containing .txt files with sentences extrated from novels to be included in the
#       testing process
# ner_model_dir_path - path to directory containing fine-tune NER model to be tested; if None standard spacy NER
#       model is used
def main(titles_path, names_gold_standard_dir_path,
         testing_data_dir_path, generated_data_dir, library='spacy', ner_model_dir_path=None):
    titles = read_file_to_list(titles_path)
    generate_generalized_data(
        titles,
        names_gold_standard_dir_path,
        generated_data_dir)

    model = load_model(library, ner_model_dir_path)

    for title in titles:
        test_data = read_sentences_from_file(
            os.path.join(testing_data_dir_path, title))
        ner_result = model.get_ner_results(test_data)

        path = os.path.join(
            generated_data_dir,
            "ner_model_annotated",
            title + '.json')

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        with open(path, 'w+') as result:
            json.dump(ner_result, result)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
