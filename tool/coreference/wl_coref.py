import sys
sys.path.append('resources/wl-coref/')

from tool.coreference.coreference_model import CoreferenceModel
from coref import CorefModel
from coref.tokenizer_customization import *
from predict import build_doc
import spacy
import torch

class WLCoref(CoreferenceModel):
    def __init__(self, model_name):
        super().__init__()

        self.model = CorefModel("resources/wl-coref/config.toml", model_name)
        self.model.load_weights(path=None, map_location="cpu",
                           ignore={"bert_optimizer", "general_optimizer",
                                   "bert_scheduler", "general_scheduler"})
        self.model.training = False
        self.spacy_tokenizer = spacy.load('en_core_web_sm')


    def get_clusters(self, text):
        doc_spacy = self.spacy_tokenizer(text)
        doc = {'cased_words':  [str(w) for s in list(doc_spacy.sents) for w in s],
               "sent_id": [s_id for s_id, s in enumerate(list(doc_spacy.sents)) for e in range(len(s))],
               "document_id": "mz_document"}
        doc = build_doc(doc, self.model)

        with torch.no_grad():
            result = self.model.run(doc)
            doc["span_clusters"] = result.span_clusters
            doc["word_clusters"] = result.word_clusters
        conll_lines = write_conll(doc)
        return conll_lines


def write_conll(output):

    results = []
    for token_id, token in enumerate(output['cased_words']):
        line = ['title', '0', str(token_id), token, '']
        results.append(line)

    for cluster_id, cluster in enumerate(output['span_clusters']):
        cluster = list(set(cluster))
        for mention in cluster:
            token_start = mention[0]
            token_end = mention[1]-1
            if token_start == token_end:
                results[token_start][4] += '(' + str(cluster_id) + ')' + '|'
            else:
                results[token_start][4] += '(' + str(cluster_id) + '|'
                results[token_end][4] += str(cluster_id) + ')' + '|'

    for token_id in range(len(results)):
        if results[token_id][4] == '':
            results[token_id][4] = '-'
        else:
            results[token_id][4] = results[token_id][4][:-1]
        results[token_id] = '\t'.join(results[token_id])

    return results
