import nltk

from tool.model.ner_model import NERModel


class NLTKModel(NERModel):

    def __init__(self, save_personal_titles, fix_personal_titles):

        super().__init__(save_personal_titles, fix_personal_titles)
        self.logger.info('NLTK model loaded.')

    def get_doc_entities(self, text, prefix=0):
        entities = []
        offset = 0
        spans = []

        for token in nltk.word_tokenize(text):
            offset = text.find(token, offset)
            spans.append((offset, offset + len(token)))

        for chunk_id, chunk in enumerate(nltk.ne_chunk(
                nltk.pos_tag(nltk.word_tokenize(text)))):
            if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                try:
                    start, end = spans[chunk_id]
                    ent_text = text[start:end]
                    self.logger.debug("ENTITY FOUND: " + ent_text)
                    if self.fix_personal_titles and ent_text.startswith(
                            self.personal_titles) and len(ent_text.split()) > 1:
                        start += (1 + len(ent_text.split(' ')[0]))
                        self.logger.debug(
                            "ENTITY WITHOUT TITLE: " + text[start:end])
                    if self.save_personal_titles:
                        personal_title = self.recognize_personal_title(
                            text, chunk_id)
                        entities.append([start+prefix, end+prefix, "PERSON", personal_title])
                    else:
                        entities.append([start+prefix, end+prefix, "PERSON"])
                except IndexError:
                    pass

        return text, entities

    def recognize_personal_title(self, text, chunk_id):
        personal_title = None
        if chunk_id > 0:
            word_before_name = nltk.word_tokenize(text)[chunk_id - 1]
            if word_before_name in self.personal_titles:
                personal_title = word_before_name
        return personal_title
