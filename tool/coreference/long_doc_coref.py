import os
import sys
sys.path.append('resources/long-doc-coref/src')

from tool.coreference.coreference_model import CoreferenceModel
from inference.inference import Inference


class LongDocCoref(CoreferenceModel):
    def __init__(self, model_name):
        super().__init__()
        print(os.getcwd())
        self.model = Inference(os.path.join("./resources/long-doc-coref/models", model_name, 'model.pth'))

    def get_clusters(self, doc):
        output = self.model.perform_coreference(doc)
        print(output)
        conll_lines = write_conll(output)
        return conll_lines


def write_conll(output):

    results = []
    # if len(output['tokenized_doc']['sentences']) != 1:
    #     print(output['tokenized_doc']['sentences'])
    #     print(output['clusters'])
    output['tokenized_doc']['sentences'] = [word for sentence in output['tokenized_doc']['sentences'] for word in sentence]
    previous_token_id = 0
    previous_token = ''
    for token_id, token in zip(output['tokenized_doc']['subtoken_map'], output['tokenized_doc']['sentences']):
        if previous_token_id == token_id:
            previous_token += token.replace('#', '')
        else:
            line = ['title', '0', str(token_id), previous_token, '']
            print(line)
            results.append(line)
            previous_token = token
            previous_token_id = token_id

    print(len(results))
    for cluster_id, cluster in enumerate(output['clusters']):
        if len(cluster) < 2:
            continue
        for mention in cluster:
            print(mention)
            token_start = mention[0][0]
            token_end = mention[0][1]
            if token_start == token_end:
                print(token_start)
                print(results[token_start])
                results[token_start][4] += '(' + str(cluster_id) + ')' + '|'
            else:
                print(token_start)
                print(results[token_start])
                results[token_start][4] += '(' + str(cluster_id) + '|'
                results[token_end][4] += str(cluster_id) + ')' + '|'

    for token_id in range(len(results)):
        if results[token_id][4] == '':
            results[token_id][4] = '-'
        else:
            results[token_id][4] = results[token_id][4][:-1]
        results[token_id] = '\t'.join(results[token_id])

    return results
