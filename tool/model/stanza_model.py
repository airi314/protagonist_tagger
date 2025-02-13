import stanza

from tool.model.ner_model import NERModel


class StanzaModel(NERModel):

    def __init__(self, model_path, save_personal_titles, fix_personal_titles):

        super().__init__(save_personal_titles, fix_personal_titles)
        self.model = stanza.Pipeline('en', processors={'ner': model_path}, tokenize_no_ssplit=True,
                                     logging_level='ERROR', use_gpu=False)
        self.logger.info('Stanza model loaded.')

    def get_doc_entities(self, text, prefix=0):
        doc = self.model(text)
        entities = []

        for index, ent in enumerate(doc.entities):
            if ent.type == "PERSON" or ent.type == "PER":
                ent_text = text[ent.start_char:ent.end_char]
                self.logger.debug("ENTITY FOUND: " + ent_text)
                if self.fix_personal_titles and ent_text.startswith(
                        self.personal_titles) and len(ent_text.split()) > 1:
                    ent.start_char += (1 + len(ent_text.split(' ')[0]))
                    self.logger.debug(
                        "ENTITY WITHOUT TITLE: " + text[ent.start_char:ent.end_char])
                if self.save_personal_titles:
                    personal_title = self.recognize_personal_title(ent, doc)
                    entities.append(
                        [ent.start_char+prefix, ent.end_char+prefix, "PERSON", personal_title])
                else:
                    entities.append([ent.start_char+prefix, ent.end_char+prefix, "PERSON"])

        return text, entities

    def recognize_personal_title(self, ent, doc):
        personal_title = None
        span_id = [x['id'] for x in doc.to_dict()[0] if x['start_char']
                   == ent.start_char][0]
        assert len(doc.sentences) == 1
        if span_id > 1:
            word_before_name = [x['text']
                                for x in doc.to_dict()[0] if x['id'] == span_id - 1][0]
            if word_before_name in self.personal_titles:
                personal_title = word_before_name
        return personal_title
