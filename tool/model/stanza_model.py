from tqdm import tqdm
import stanza

from tool.model.ner_model import NERModel


class StanzaModel(NERModel):

    def __init__(self):

        super().__init__()
        self.model = stanza.Pipeline('en', processors='tokenize,ner', tokenize_no_ssplit=True)

    def get_ner_results(self, data):

        results = []
        for sentence in tqdm(data):
            doc = self.model(sentence)
            entities = []
            for index, ent in enumerate(doc.entities):
                if ent.type == "PERSON":
                    entities.append([ent.start_char, ent.end_char, "PERSON"])
            results.append({'content': sentence, 'entities': entities})

        return results
