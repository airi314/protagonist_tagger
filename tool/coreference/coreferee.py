import spacy

from tool.coreference.coreference_model import CoreferenceModel


class Coreferee(CoreferenceModel):
    def __init__(self, model_name):
        super().__init__()
        self.model = spacy.load(model_name)
        self.model.add_pipe('coreferee')

    def get_clusters(self, text):
        doc = self.model(text)
        conll_lines = write_conll(doc)
        return conll_lines


def write_conll(doc):
    tokens_clusters = {}
    for cluster_id, cluster in enumerate(doc._.coref_chains):
        for mention in cluster:
            tokens_clusters[mention[0]] = cluster_id

    results = []
    for token_id, token in enumerate(doc):
        mention = '-'
        if token_id in tokens_clusters.keys():
            mention = '(' + str(tokens_clusters[token_id]) + ')'
        line = ['title', '0', str(token_id), token.text, mention]
        results.append('\t'.join(line))

    return results
