from tool.model.ner_model import NERModel
import spacy
from spacy.tokens import Span
from tqdm import tqdm

class SpacyModel(NERModel):

    def __init__(self, model_path):

        self.model_path = model_path
        self.model = spacy.load(model_path, enable='ner')

    def get_ner_results(self, data):

        results = []
        for sentence in tqdm(data):
            doc = self.model(sentence)
            entities = []
            for index, ent in enumerate(doc.ents):
                if ent.label_ == "PERSON":
                    span = Span(doc, ent.start, ent.end, label="PERSON")
                    doc.ents = [span if e == ent else e for e in doc.ents]
                    entities.append([ent.start_char, ent.end_char, "PERSON"])

            results.append({'content': sentence, 'entities': entities})

        return results

    def train_model(self):

        pass

