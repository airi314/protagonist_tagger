import torch
import spacy
from predict import build_doc
from coref.tokenizer_customization import *
from coref import CorefModel
from tool.coreference.coreference_model import CoreferenceModel
import sys
sys.path.append('resources/wl-coref/')


class WLCoref(CoreferenceModel):
    def __init__(self, model_name, save_singletons):
        super().__init__()

        self.model = CorefModel("resources/wl-coref/config.toml", model_name)
        self.model.load_weights(path=None, map_location="cpu",
                                ignore={"bert_optimizer", "general_optimizer",
                                        "bert_scheduler", "general_scheduler"})
        self.model.training = False
        self.spacy_tokenizer = spacy.load('en_core_web_sm')
        self.save_singletons = save_singletons

    def get_clusters(self, text, start_cluster_id=None, save_conll=False):
        doc_spacy = self.spacy_tokenizer(text)
        doc = {'cased_words': [str(w).strip() for s in list(doc_spacy.sents) for w in s],
               "sent_id": [s_id for s_id, s in enumerate(list(doc_spacy.sents)) for e in range(len(s))],
               "document_id": "mz_document"}
        doc = build_doc(doc, self.model)
        with torch.no_grad():
            try:
                result = self.model.run(doc)
                doc["span_clusters"] = result.span_clusters
                doc["word_clusters"] = result.word_clusters
            except AssertionError as error:
                print(error, doc)
                doc["span_clusters"] = []
                doc["word_clusters"] = []

        if save_conll:
            lines, start_cluster_id = self.write_conll(doc, start_cluster_id)
            return lines, start_cluster_id

    def write_conll(self, output, start_cluster_id):

        results = []
        for token_id, token in enumerate(output['cased_words']):
            line = ['title', '0', str(token_id), token, '']
            results.append(line)

        print('Number of clusters:', len(output['span_clusters']))
        cluster_id = None
        if len(output['span_clusters']) != 1 or len(
                output['span_clusters'][0]) < 50:
            for cluster_id, cluster in enumerate(output['span_clusters']):
                cluster = list(set(cluster))
                if not self.save_singletons and len(cluster) < 2:
                    continue
                else:
                    for mention in cluster:
                        token_start = mention[0]
                        token_end = mention[1] - 1
                        if token_start == token_end:
                            results[token_start][4] += '(' + str(
                                start_cluster_id + cluster_id) + ')' + '|'
                        else:
                            results[token_start][4] += '(' + str(
                                start_cluster_id + cluster_id) + '|'
                            results[token_end][4] += str(
                                start_cluster_id + cluster_id) + ')' + '|'

        for token_id in range(len(results)):
            if results[token_id][4] == '':
                results[token_id][4] = '-'
            else:
                results[token_id][4] = results[token_id][4][:-1]
            results[token_id] = '\t'.join(results[token_id])

        return results, start_cluster_id + cluster_id if cluster_id else start_cluster_id
