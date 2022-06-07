import os
import sys
sys.path.append('resources/fast-coref/src')

from tool.coreference.coreference_model import CoreferenceModel
from inference.model_inference import Inference


class FastCoref(CoreferenceModel):
    def __init__(self, model_name):
        super().__init__()
        print(os.getcwd())
        self.model = Inference(os.path.join("./resources/fast-coref/models", model_name),
                               encoder_name="shtoshni/longformer_coreference_ontonotes")

    def get_clusters(self, doc):
        output = self.model.perform_coreference(doc)
        conll_lines = write_conll(output)
        return conll_lines


def write_conll(output):

    results = []
    for token_id, token in enumerate(output['tokenized_doc']['orig_tokens']):
        line = ['title', '0', str(token_id), token, '']
        results.append(line)

    for cluster_id, cluster in enumerate(output['clusters']):
        if len(cluster) < 2:
            continue
        for mention in cluster:
            token_start = output['tokenized_doc']['subtoken_map'][mention[0][0]]
            token_end = output['tokenized_doc']['subtoken_map'][mention[0][1]]
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
