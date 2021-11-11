from tqdm import tqdm
from flair.models import SequenceTagger
from flair.data import Sentence

from tool.model.ner_model import NERModel


class FlairModel(NERModel):

    def __init__(self, model_path, save_personal_titles):

        super().__init__(save_personal_titles)
        self.model = SequenceTagger.load(model_path)

    def get_ner_results(self, data):

        results = []
        for sentence in tqdm(data):
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
        print(ent)
        print(doc)
        return ""

    # def recognize_personal_title(self, doc, index):
    #     personal_title = None
    #     span = doc.ents[index]
    #     if span.start > 0:
    #         word_before_name = doc[span.start - 1].text
    #         if word_before_name.replace(".", "") in self.personal_titles:
    #             personal_title = word_before_name.replace(".", "")
    #         if word_before_name.lower() == "the":
    #             personal_title = "the"
    #
    #     return personal_title
