from tqdm import tqdm
import nltk

from tool.model.ner_model import NERModel


class NLTKModel(NERModel):

    def __init__(self):

        super().__init__()
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('maxent_ne_chunker')
        nltk.download('words')

    def get_ner_results(self, data):

        results = []
        for sentence in tqdm(data):

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
                        entities.append([span[0], span[1], "PERSON"])
                    except IndexError:
                        pass
            results.append({'content': sentence, 'entities': entities})

        return results
