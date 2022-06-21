import flair
from flair.models import SequenceTagger
from flair.data import Sentence
import torch

from tool.model.ner_model import NERModel


class FlairModel(NERModel):

    def __init__(self, model_path, save_personal_titles, fix_personal_titles):

        super().__init__(save_personal_titles, fix_personal_titles)
        flair.device = torch.device("cpu")
        self.model = SequenceTagger.load(model_path)
        self.logger.info('Flair model "' + model_path + '" loaded.')

    def get_doc_entities(self, text, prefix=0):
        doc = Sentence(text)
        self.model.predict(doc)

        entities = []
        for ent in doc.get_spans("ner"):
            if ent.labels[0].to_dict()["value"] == "PER":
                ent_id = ent[0].idx
                if hasattr(ent, 'start_pos'):
                    start, end = ent.start_pos, ent.end_pos
                else:
                    start, end = ent.start_position, ent.end_position
                ent_text = text[start:end]
                self.logger.debug("ENTITY FOUND: " + ent_text)
                if self.fix_personal_titles and ent_text.startswith(
                        self.personal_titles) and len(ent_text.split()) > 1:
                    start += (1 + len(ent_text.split(' ')[0]))
                    ent_id += 1
                    self.logger.debug(
                        "ENTITY WITHOUT TITLE: " + text[start:end])
                if start < end:
                    if self.save_personal_titles:
                        personal_title = self.recognize_personal_title(
                            ent_id, doc)
                        entities.append(
                            [start+prefix, end+prefix, "PERSON", personal_title])
                    else:
                        entities.append([start+prefix, end+prefix, "PERSON"])

        return text, entities

    def recognize_personal_title(self, ent_id, doc):
        personal_title = None
        token_id = ent_id - 1
        if token_id > 0:
            word_before_name = doc.tokens[token_id - 1].text
            if word_before_name in self.personal_titles:
                personal_title = word_before_name
            return personal_title
