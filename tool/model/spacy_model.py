from tqdm import tqdm
import spacy

from tool.model.ner_model import NERModel


class SpacyModel(NERModel):

    def __init__(self, model_path, save_personal_titles):
        super().__init__(save_personal_titles)
        self.model = spacy.load(model_path, enable='ner')
        print('Spacy model "' + model_path + '" loaded.')

    def get_ner_results(self, data):

        results = []
        for sentence in tqdm(data):
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

    def recognize_personal_title(self, doc, index):
        personal_title = None
        span = doc.ents[index]
        if span.start > 0:
            word_before_name = doc[span.start - 1].text
            if word_before_name.replace(".", "") in self.personal_titles:
                personal_title = word_before_name.replace(".", "")
            if word_before_name.lower() == "the":
                personal_title = "the"

        return personal_title
