from tqdm import tqdm
from flair.models import SequenceTagger
from flair.data import Sentence

from tool.model.ner_model import NERModel


class FlairModel(NERModel):

    def __init__(self, model_path):

        super().__init__()
        self.model = SequenceTagger.load(model_path)

    def get_ner_results(self, data):

        results = []
        for sentence in tqdm(data):
            doc = Sentence(sentence)
            self.model.predict(doc)

            entities = []
            for ent in doc.get_spans('ner'):
                if ent.labels[0].to_dict()['value'] == 'PER':
                    entities.append([ent.start_pos, ent.end_pos, "PERSON"])
            results.append({'content': sentence, 'entities': entities})

        return results
