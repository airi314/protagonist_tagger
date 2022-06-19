from fuzzywuzzy import fuzz
import itertools
import logging
from tqdm import tqdm

from tool.gender_checker import get_name_gender, get_personal_titles, \
    create_titles_and_gender_dictionary
import gender_guesser.detector as gender_guesser
from tool.diminutives_recognizer import get_names_from_diminutive, \
    create_diminutives_dictionary
from tool.model.utils import load_model


class NamesMatcher:
    def __init__(self, partial_ratio_precision, library, model_path,
                 rules, fix_personal_titles):
        self.personal_titles = get_personal_titles()
        self.titles_gender_dict = create_titles_and_gender_dictionary()
        self.model = load_model(library, model_path, True, fix_personal_titles)
        self.partial_ratio_precision = partial_ratio_precision
        self.logger = logging.getLogger()
        self.rules = rules
        self.gender_detector = gender_guesser.Detector()
        self.diminutives_dictionary = create_diminutives_dictionary()

    def recognize_person_entities(self, ner_results, characters):
        matcher_results = []
        for sent_id, sent_result in enumerate(tqdm(ner_results, leave=False)):
            self.logger.debug(
                "SENTENCE " +
                str(sent_id) +
                ": " +
                sent_result["content"])
            entities = []
            for (ent_start, ent_stop, ent_label,
                 personal_title) in sent_result["entities"]:
                person = sent_result["content"][ent_start:ent_stop]
                self.logger.debug("ENTITY TO MATCH: " + person)
                final_match = self.find_match_for_person(
                    person, personal_title, characters)
                self.logger.debug("FINAL MATCH: " + final_match)
                entities.append([ent_start, ent_stop, final_match])

            self.logger.debug("ENTITIES: " + str(entities))
            matcher_results.append(
                {"content": sent_result["content"], "entities": entities})

        return matcher_results

    def find_match_for_person(self, person, personal_title, characters):
        potential_matches = []
        for index, character in enumerate(characters):
            ratio = fuzz.ratio(person, character)
            ratio_title = fuzz.ratio(
                ((personal_title + " ") if personal_title is not None else "") + person,
                character)
            partial_ratio = get_partial_ratio_for_all_permutations(
                person, character)
            if ratio == 100 or ratio_title == 100:
                return character
            elif partial_ratio >= self.partial_ratio_precision:
                potential_matches.append([character, partial_ratio])

        potential_matches = sorted(
            potential_matches,
            key=lambda x: x[1],
            reverse=True)

        self.logger.debug("PERSONAL TITLE: " + str(personal_title or "None"))
        self.logger.debug("POTENTIAL MATCHES: " + str(potential_matches))
        final_match = self.choose_best_match(person, personal_title,
                                             potential_matches, characters)

        if final_match is None:
            return "PERSON"
        return final_match

    def choose_best_match(self, person, personal_title,
                          potential_matches, characters):
        self.logger.debug("choosing best match...")
        final_match = None
        if len(potential_matches):
            final_match = self.handle_multiple_potential_matches(
                person, personal_title, potential_matches)

        if self.rules["check_diminutive"] and not final_match:
            potential_names_from_diminutive = get_names_from_diminutive(
                person.split()[0], self.diminutives_dictionary)
            if potential_names_from_diminutive:
                self.logger.debug(
                    "DIMINUTIVE: " +
                    str(potential_names_from_diminutive))
                for character in characters:
                    for name in potential_names_from_diminutive:
                        if name in character.lower().split():
                            return character
        return final_match

    def handle_multiple_potential_matches(
            self, person, personal_title, potential_matches):

        title_gender = None
        if self.rules["match_personal_title_100"] and personal_title:
            title_gender = self.titles_gender_dict[personal_title][0]
            for match in potential_matches:
                if match[1] == 100 and personal_title in match[0]:
                    self.logger.debug(
                        "FINAL MATCH BY PERSONAL TITLE: " + match[0])
                    return match[0]

        entity_gender = get_name_gender(person, self.gender_detector)
        self.logger.debug("TITLE GENDER: " + str(title_gender or "None"))
        self.logger.debug("ENTITY GENDER: " + str(entity_gender or "None"))
        for match in potential_matches:
            self.logger.debug("POTENTIAL MATCH: " + match[0])
            character_gender = get_name_gender(match[0], self.gender_detector)
            self.logger.debug("CHARACTER GENDER: " +
                              str(character_gender or "None"))
            character_title = None

            if match[0].split()[0] in self.personal_titles:
                character_title = match[0].split()[0].lower()
                self.logger.debug("CHARACTER TITLE: " + character_title)

            if self.rules["match_personal_title"] and \
                    character_title and personal_title and character_title != personal_title:
                self.logger.debug(
                    "character title different from entity title")
                continue

            if self.rules["match_title_gender"] and \
                    character_gender and title_gender and character_gender != title_gender:
                self.logger.debug(
                    "character gender different from title gender")
                continue

            if self.rules["match_entity_gender"] and \
                    not title_gender and character_gender and entity_gender and character_gender != entity_gender:
                self.logger.debug(
                    "character gender different from entity gender")
                continue

            return match[0]
        return None


def get_partial_ratio_for_all_permutations(potential_match, character_name):
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
