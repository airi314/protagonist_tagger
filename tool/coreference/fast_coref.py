import os
import sys
sys.path.append('resources/fast-coref/src')

from tool.coreference.coreference_model import CoreferenceModel
from inference.model_inference import Inference


class FastCoref(CoreferenceModel):
    def __init__(self, model_name, save_singletons):
        super().__init__()
        self.model = Inference(os.path.join("./resources/fast-coref/models", model_name),
                               encoder_name="shtoshni/longformer_coreference_ontonotes")
        self.model.device = 'cpu'
        self.save_singletons = save_singletons

    def get_clusters(self, doc, start_cluster_id = None, save_conll = False):
        output = self.model.perform_coreference(doc)
        print('doc', doc[:200])
        if save_conll:
            return self.write_conll(output, start_cluster_id)
        else:
            return self.write_json(doc, output)

    def write_conll(self, output, start_cluster_id):

        results = []
        for token_id, token in enumerate(output['tokenized_doc']['orig_tokens']):
            line = ['title', '0', str(token_id), token, '']
            results.append(line)

        cluster_id = None
        for cluster_id, cluster in enumerate(output['clusters']):
            if self.save_singletons and len(cluster) < 2:
                continue
            else:
                for mention in cluster:
                    token_start = output['tokenized_doc']['subtoken_map'][mention[0][0]]
                    token_end = output['tokenized_doc']['subtoken_map'][mention[0][1]]
                    if token_start == token_end:
                        results[token_start][4] += '(' + str(start_cluster_id+cluster_id) + ')' + '|'
                    else:
                        results[token_start][4] += '(' + str(start_cluster_id+cluster_id) + '|'
                        results[token_end][4] += str(start_cluster_id+cluster_id) + ')' + '|'

        for token_id in range(len(results)):
            if results[token_id][4] == '':
                results[token_id][4] = '-'
            else:
                results[token_id][4] = results[token_id][4][:-1]
            results[token_id] = '\t'.join(results[token_id])

        return results, start_cluster_id+cluster_id if cluster_id else start_cluster_id


    def write_json(self, doc, output):
        results = {'content': doc, 'clusters': []}
        print('content', results['content'][:200])
        output['tokenized_doc']['spans'] = self.get_spans(doc, output)
        for cluster in output['clusters']:
            # print('new cluster', cluster)
            cluster_positions = []
            if not self.save_singletons and len(cluster) < 2:
                continue
            else:
                for mention in cluster:
                    token_start, token_end = mention[0]
                    orig_token_start = output['tokenized_doc']['subtoken_map'][token_start]
                    orig_token_end = output['tokenized_doc']['subtoken_map'][token_end]
                    start_position = output['tokenized_doc']['spans'][orig_token_start][0]
                    end_position = output['tokenized_doc']['spans'][orig_token_end][1]
                    # if not doc[start_position:end_position] == mention[1]:
                    #     print(doc[start_position:end_position], mention[1])
                    # delay_end = 0
                    # while doc[start_position:end_position] != mention[1]:
                    #     print(doc[start_position:end_position])
                    #     if mention[1].startswith(doc[start_position:end_position]):
                    #         delay_end += 1
                    #         end_position = tokenized_doc.token_to_chars(orig_token_end + delay + delay_end).end
                    #     else:
                    #         delay += 1
                    #         start_position = tokenized_doc.token_to_chars(orig_token_start + delay).start
                    #         end_position = tokenized_doc.token_to_chars(orig_token_end + delay).end
                    # print(doc[start_position:end_position])
                    cluster_positions.append([start_position, end_position])
                results['clusters'].append(cluster_positions)
        return results

    def get_spans(self, doc, output):
        position = 0
        token_spans = []
        for token in output['tokenized_doc']['orig_tokens']:
            # print(token)
            start_position = doc[position:].find(token)
            token_spans.append([position + start_position,
                                position + start_position + len(token)])
            position = position + start_position + len(token)
            # print(position + start_position, position + start_position + len(token))
        return token_spans

