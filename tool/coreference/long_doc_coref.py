from inference.inference import Inference
from tool.coreference.coreference_model import CoreferenceModel
import os
import sys
sys.path.append('resources/long-doc-coref/src')


class LongDocCoref(CoreferenceModel):
    def __init__(self, model_name, save_singletons):
        super().__init__()
        self.model = Inference(
            os.path.join(
                "./resources/long-doc-coref/models",
                model_name,
                'model.pth'))
        self.save_singletons = save_singletons

    def get_clusters(self, doc, start_cluster_id=None, save_conll=False):
        output = self.model.perform_coreference(doc)
        if save_conll:
            return self.write_conll(output, start_cluster_id)
        else:
            return self.write_json(doc, output)

    def write_conll(self, output, start_cluster_id):
        results = []
        output['tokenized_doc']['sentences'] = [
            word for sentence in output['tokenized_doc']['sentences'] for word in sentence]
        previous_token_id = 0
        previous_token = ''
        for token_id, token in zip(
                output['tokenized_doc']['subtoken_map'], output['tokenized_doc']['sentences']):
            if previous_token_id == token_id:
                previous_token += token.replace('#', '')
            else:
                line = ['title', '0', str(token_id), previous_token, '']
                results.append(line)
                previous_token = token
                previous_token_id = token_id
        results.append(
            ['title', '0', str(previous_token_id), previous_token, ''])
        cluster_id = None
        for cluster_id in range(len(output['clusters'])):
            cluster = output['clusters'][cluster_id]
            if not self.save_singletons and len(cluster) < 2:
                continue
            elif cluster_id > 0 and cluster[0][0] == output['clusters'][cluster_id - 1][0][0]:
                continue
            else:
                for mention in cluster:
                    token_start = mention[0][0]
                    token_end = mention[0][1]
                    if token_start == token_end:
                        results[token_start][4] += '(' + str(
                            start_cluster_id + cluster_id) + ')' + '|'
                    else:
                        results[token_start][4] += '(' + \
                            str(start_cluster_id + cluster_id) + '|'
                        results[token_end][4] += str(
                            start_cluster_id + cluster_id) + ')' + '|'

        for token_id in range(len(results)):
            if results[token_id][4] == '':
                results[token_id][4] = '-'
            else:
                results[token_id][4] = results[token_id][4][:-1]
            results[token_id] = '\t'.join(results[token_id])

        return results, start_cluster_id + cluster_id if cluster_id else start_cluster_id

    def write_json(self, doc, output):
        pass
