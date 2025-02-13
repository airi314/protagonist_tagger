import spacy

from tool.model.ner_model import NERModel


class SpacyModel(NERModel):

    def __init__(self, model_path, save_personal_titles, fix_personal_titles):
        super().__init__(save_personal_titles, fix_personal_titles)
        self.model = spacy.load(model_path)
        self.logger.info('Spacy model "' + model_path + '" loaded.')

    def get_doc_entities(self, text, prefix=0):
        doc = self.model(text)
        entities = []

        for index, ent in enumerate(doc.ents):
            if ent.label_ == "PERSON":
                start, end = ent.start_char, ent.end_char
                ent_text = text[start:end]
                self.logger.debug("ENTITY FOUND: " + ent_text)
                if self.fix_personal_titles and ent_text.startswith(
                        self.personal_titles) and len(ent_text.split()) > 1:
                    start += (1 + len(ent_text.split(' ')[0]))
                    self.logger.debug(
                        "ENTITY WITHOUT TITLE: " + text[start:end])
                if self.save_personal_titles:
                    personal_title = self.recognize_personal_title(doc, index)
                    entities.append([start+prefix, end+prefix, "PERSON", personal_title])
                else:
                    entities.append([start+prefix, end+prefix, "PERSON"])

        return text, entities

    def recognize_personal_title(self, doc, index):
        personal_title = None
        span = doc.ents[index]
        if span.start > 0:
            word_before_name = doc[span.start - 1].text
            if word_before_name.replace(".", "") in self.personal_titles:
                personal_title = word_before_name.replace(".", "")
        return personal_title
