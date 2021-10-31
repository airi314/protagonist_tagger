from tool.model.ner_model import NERModel
import spacy
from tqdm import tqdm

class SpacyModel(NERModel):

    def __init__(self, model_path, save_personal_titles):

        super().__init__(save_personal_titles)
        self.model_path = model_path
        self.model = spacy.load(model_path, enable='ner')

    def get_ner_results(self, data):

        results = []
        for sentence in data:
            doc = self.model(sentence)
            entities = []
            for index, ent in enumerate(doc.ents):
                if ent.label_ == "PERSON":
                    if self.save_personal_titles:
                        personal_title = self.recognize_personal_title(doc, index)
                        entities.append([ent.start_char, ent.end_char, "PERSON", personal_title])
                    else:
                        entities.append([ent.start_char, ent.end_char, "PERSON"])

            results.append({'content': doc.text, 'entities': entities})

        return results

    def train_model(self):

        pass

    def recognize_personal_title(self, doc, index):
        personal_title = None
        span = doc.ents[index]
        if span.start > 0:
            word_before_name = doc[span.start - 1]
            if word_before_name.text.replace(".", "") in self.personal_titles:
                personal_title = word_before_name.text.replace(".", "")
            if word_before_name.text.lower() == "the":
                personal_title = "the"

        return personal_title
