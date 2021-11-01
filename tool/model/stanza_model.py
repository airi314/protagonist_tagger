import stanza

from tool.model.ner_model import NERModel


class StanzaModel(NERModel):

    def __init__(self, save_personal_titles):

        super().__init__(save_personal_titles)
        self.model = stanza.Pipeline('en', processors='tokenize,ner', tokenize_no_ssplit=True)

    def get_ner_results(self, data):

        results = []
        for sentence in data:
            doc = self.model(sentence)
            entities = []
            for index, ent in enumerate(doc.entities):
                if ent.type == "PERSON":
                    if self.save_personal_titles:
                        personal_title = self.recognize_personal_title(ent, doc)
                        entities.append([ent.start_char, ent.end_char, "PERSON", personal_title])
                    else:
                        entities.append([ent.start_char, ent.end_char, "PERSON"])
            results.append({'content': sentence, 'entities': entities})
        return results

    def recognize_personal_title(self, ent, doc):
        personal_title = None
        span_id = [x['id'] for x in doc.sentences[0].tokens if x['start_char'] == ent.start_char][0]
        assert len(doc.sentences) == 1
        if span_id > 1:
            word_before_name = [x['text'] for x in doc.sentences[0].tokens if x['id'] == span_id - 1][0]
            if word_before_name.replace(".", "") in self.personal_titles:
                personal_title = word_before_name.replace(".", "")
            if word_before_name.lower() == "the":
                personal_title = "the"

        return personal_title
