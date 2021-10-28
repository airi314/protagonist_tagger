from tool.model.ner_model import NERModel
import stanza
from tqdm import tqdm

class StanzaModel(NERModel):

    def __init__(self):

        self.model = stanza.Pipeline('en', processors = 'tokenize,ner')

    def get_ner_ results(self, data):

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