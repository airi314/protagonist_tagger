import json
from fuzzywuzzy import fuzz
import itertools
import os

from tool.file_and_directory_management import write_list_to_file
from tool.gender_checker import get_name_gender, get_personal_titles, create_titles_and_gender_dictionary
from tool.diminutives_recognizer import get_names_from_diminutive
from tool.model.utils import load_model


def prepare_list_for_ratios(characters):
    ratios = []
    row = ["RECOGNIZED NAMED ENTITY", "MATCH"]
    for char in characters:
        row.append(char)
    ratios.append(row)

    return ratios


class NamesMatcher:
    def __init__(self, partial_ratio_precision, library="spacy", model_path="en_core_web_sm"):
        self.personal_titles = get_personal_titles()
        self.titles_gender_dict = create_titles_and_gender_dictionary()
        self.model = load_model(library, model_path, True)
        self.partial_ratio_precision = partial_ratio_precision

    def recognize_person_entities(self, text, characters):

        matches_table = prepare_list_for_ratios(characters)
        ner_results = self.model.get_ner_results(text)
        matcher_results = []

        for sentence in ner_results:
            entities = []
            for (ent_start, ent_stop, ent_label, personal_title) in sentence['entities']:
                person = sentence['content'][ent_start:ent_stop]
                row = self.find_match_for_person(person, personal_title, characters)

                if row is not None:
                    matches_table.append(row)
                    if row[1] is not None:
                        entities.append([ent_start, ent_stop, row[1]])

            matcher_results.append({'content': sentence["content"], 'entities': entities})

        return matches_table, matcher_results

    def match_names_for_text(self, characters, text, results_dir, filename=None, save_ratios=False):

        matches_table, train_data = self.recognize_person_entities(text, characters)

        if filename is not None:

            if save_ratios:
                write_list_to_file(
                    os.path.join(
                        results_dir,
                        "ratios",
                        filename),
                    matches_table)

            with open(os.path.join(results_dir, filename + '.json'), 'w', encoding='utf8') as result:
                json.dump(train_data, result, ensure_ascii=False)

        return matches_table

    def matcher_test(self, characters, testing_string,
                     results_dir, filename=None):
        matches_table, train_data = self.recognize_person_entities(testing_string, characters)

        if filename is not None:
            write_list_to_file(
                os.path.join(
                    results_dir,
                    "ratios",
                    filename +
                    "_test"),
                matches_table)
            with open(os.path.join(results_dir, filename + "_test_spacy.json"), 'w') as result:
                json.dump(train_data, result)

    def find_match_for_person(self, person, personal_title, characters):
        row_ratios = []
        potential_matches = []
        if "Miss " in person:
            person = person.replace("Miss ", "")
            personal_title = "Miss"

        for index, char in enumerate(characters):
            # partial_ratio = fuzz.partial_ratio(((personal_title + " ")
            # if personal_title is not None else "") + person, char)
            # ratio = fuzz.ratio(((personal_title + " ") if personal_title is not None else "") + person, char)
            partial_ratio = self.get_partial_ratio_for_all_permutations(
                person, char)
            ratio = fuzz.ratio(
                ((personal_title.replace(
                    ".",
                    "") +
                  " ") if personal_title is not None else "") +
                person,
                char)
            ratio_no_title = fuzz.ratio(person, char)
            if ratio == 100 or ratio_no_title == 100:
                potential_matches = [[char, ratio]]
                row_ratios = row_ratios + ["---" for i in range(0, len(characters) - index)]
                break
            if partial_ratio >= self.partial_ratio_precision:
                row_ratios.append("---" + str(partial_ratio) + "---")
                potential_matches.append([char, partial_ratio])
            else:
                row_ratios.append(str(partial_ratio))

        potential_matches = sorted(
            potential_matches,
            key=lambda x: x[1],
            reverse=True)
        row = [
            ((personal_title + " ") if personal_title is not None else "") + str(person)]
        ner_match = self.choose_best_match(
            person, personal_title, potential_matches, characters)
        if ner_match is None:
            return None

        row.append(ner_match)
        row.extend(row_ratios)

        return row

    def get_partial_ratio_for_all_permutations(
            self, potential_match, character_name):
        character_name_components = character_name.split()
        character_name_permutations = list(
            itertools.permutations(character_name_components))
        partial_ratios = []
        for permutation in character_name_permutations:
            partial_ratios.append(
                fuzz.partial_ratio(
                    ' '.join(permutation),
                    potential_match))

        return max(partial_ratios)

    def choose_best_match(self, person, personal_title,
                          potential_matches, characters):
        if len(potential_matches) > 1:
            ner_match = self.handle_multiple_potential_matches(
                person, personal_title, potential_matches)
        elif len(potential_matches) == 1:
            ner_match = potential_matches[0][0]
        else:
            ner_match = "PERSON"
            potential_names_from_diminutive = get_names_from_diminutive(person)
            if potential_names_from_diminutive is not None:
                for char in characters:
                    for name in potential_names_from_diminutive:
                        if name in char.lower().split():
                            return char

        return ner_match

    def handle_multiple_potential_matches(
            self, person, personal_title, potential_matches):
        ner_match = None
        if personal_title is not None:
            if personal_title == "the":
                return "the " + person
            else:
                title_gender = self.titles_gender_dict[personal_title][0]
                for match in potential_matches:
                    if get_name_gender(match[0]) == title_gender:
                        ner_match = match[0]
                        break

        else:  # todo handle Bennet sisters, daughters, etc.
            ner_match = potential_matches[0][0]

        return ner_match
