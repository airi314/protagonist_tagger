from tool.gender_checker import get_personal_titles
from tqdm import tqdm
import logging


class NERModel:
    def __init__(self, save_personal_titles, fix_personal_titles):
        self.save_personal_titles = save_personal_titles
        self.personal_titles = tuple(get_personal_titles())
        self.fix_personal_titles = fix_personal_titles
        self.logger = logging.getLogger()

    def get_ner_results(self, data, full_text=False):
        if full_text:
            entities = []
            prefix = 0
            for text_part_id, text_part in enumerate(tqdm(data, leave=False)):
                self.logger.debug("TEXT PART " + str(text_part_id) + ": " + text_part)
                text, entities_part = self.get_doc_entities(text_part, prefix)
                self.logger.debug("ENTITIES: " + str(entities))
                prefix += len(text_part) + 1
                entities += entities_part
            results = [{"content": '\n'.join(data), "entities": entities}]
        else:
            results = []
            for sent_id, sentence in enumerate(tqdm(data, leave=False)):
                self.logger.debug("SENTENCE " + str(sent_id) + ": " + sentence)
                text, entities = self.get_doc_entities(sentence)
                self.logger.debug("ENTITIES: " + str(entities))
                results.append({"content": text, "entities": entities})
        return results

    def get_doc_entities(self, text):
        pass
