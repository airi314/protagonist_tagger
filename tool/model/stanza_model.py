import stanza
from tqdm import tqdm

from tool.model.ner_model import NERModel


class StanzaModel(NERModel):

    def __init__(self):

        self.model = stanza.Pipeline('en', processors='tokenize,ner')

    def get_ner_results(self, data):

        results = []
        for sentence in tqdm(data):
            doc = self.model(sentence)
            entities = []
            for ent in doc.entities:
                if ent.type == "PERSON":
                    entities.append([ent.start_char, ent.end_char, "PERSON"])
            results.append({'content': sentence, 'entities': entities})
        return results

    def train_model(self):

        pass