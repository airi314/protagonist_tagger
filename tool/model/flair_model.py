from tool.model.ner_model import NERModel
from flair.data import Sentence
from flair.models import SequenceTagger
from tqdm import tqdm


class FlairModel(NERModel):

    def __init__(self, model_path, save_personal_titles):

        super().__init__(save_personal_titles)
        self.model = SequenceTagger.load(model_path)

    def get_ner_results(self, data):

        results = []
        for sentence in data:
            doc = Sentence(sentence)
            self.model.predict(doc)

            entities = []
            for ent in doc.get_spans('ner'):
                if ent.labels[0].to_dict()['value'] == 'PER':
                    if self.save_personal_titles:
                        personal_title = self.recognize_personal_title(ent, doc)
                        entities.append([ent.start_pos, ent.end_pos, "PERSON", personal_title])
                    else:
                        entities.append([ent.start_pos, ent.end_pos, "PERSON"])

            results.append({'content': sentence, 'entities': entities})

        return results

    def recognize_personal_title(self, ent, doc):
        pass