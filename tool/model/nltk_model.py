import nltk
from tqdm import tqdm
from tool.model.ner_model import NERModel


class NLTKModel(NERModel):

    def __init__(self, save_personal_titles):

        super().__init__(save_personal_titles)
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('maxent_ne_chunker')
        nltk.download('words')

    def get_ner_results(self, data):

        results = []
        for sentence in data:

            entities = []
            offset = 0
            spans = []

            for token in nltk.word_tokenize(sentence):
                offset = sentence.find(token, offset)
                spans.append((offset, offset + len(token)))

            for chunk_id, chunk in enumerate(nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sentence)))):

                if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                    try:
                        span = spans[chunk_id]
                        if self.save_personal_titles:
                            personal_title = self.recognize_personal_title(sentence, chunk_id)
                            entities.append([span[0], span[1], "PERSON", personal_title])

                        else:
                            entities.append([span[0], span[1], "PERSON"])
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
