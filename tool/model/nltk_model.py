from tqdm import tqdm
import nltk

from tool.model.ner_model import NERModel


class NLTKModel(NERModel):

    def __init__(self, save_personal_titles, fix_personal_titles):

        super().__init__(save_personal_titles, fix_personal_titles)
        print('NLTK model loaded.')

    def get_ner_results(self, data):

        results = []
        for sentence in tqdm(data, leave=False):

            entities = []
            offset = 0
            spans = []

            for token in nltk.word_tokenize(sentence):
                offset = sentence.find(token, offset)
                spans.append((offset, offset + len(token)))

            for chunk_id, chunk in enumerate(nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sentence)))):

                if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                    try:
                        start, end = spans[chunk_id]
                        text = sentence[start:end]
                        if self.fix_personal_titles and text.startswith(self.personal_titles):
                            start += (1 + len(text.split(' ')[0]))
                        if self.save_personal_titles:
                            personal_title = self.recognize_personal_title(sentence, chunk_id)
                            entities.append([start, end, "PERSON", personal_title])
                        else:
                            entities.append([start, end, "PERSON"])
                    except IndexError:
                        pass

            results.append({'content': sentence, 'entities': entities})

        return results

    def recognize_personal_title(self, sentence, chunk_id):
        personal_title = None
        if chunk_id > 0:
            word_before_name = nltk.word_tokenize(sentence)[chunk_id - 1]
            if word_before_name.replace(".", "") in self.personal_titles:
                personal_title = word_before_name.replace(".", "")
            if word_before_name.lower() == "the":
                personal_title = "the"
        return personal_title
