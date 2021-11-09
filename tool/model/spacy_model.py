from tqdm import tqdm
import spacy

from tool.model.ner_model import NERModel


class SpacyModel(NERModel):

    def __init__(self, model_path):

        super().__init__()
        self.model = spacy.load(model_path, enable='ner')

    def get_ner_results(self, data):

        results = []
        for sentence in tqdm(data):
            doc = self.model(sentence)
            entities = []
            for index, ent in enumerate(doc.ents):
                if ent.label_ == "PERSON":
                    entities.append([ent.start_char, ent.end_char, "PERSON"])
            results.append({'content': doc.text, 'entities': entities})

        return results
