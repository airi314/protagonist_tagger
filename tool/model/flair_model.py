from tool.model.ner_model import NERModel
from flair.data import Sentence
from flair.models import SequenceTagger

class FlairModel(NERModel):

    def __init__(self, model_path):

        self.model_path = model_path
        self.model = SequenceTagger.load(self.model_path)

    def get_ner_results(self, data):

        results = []
        for sentence in data:
            doc = Sentence(sentence)
            self.model.predict(doc)

            entities = []
            for ent in doc.get_spans('ner'):
                if ent.labels[0].to_dict()['value'] == 'PER':
                    entities.append([ent.start_pos, ent.end_pos, "PERSON"])

            results.append({'content': sentence, 'entities': entities})

        return results

    def train_model(self):

        pass