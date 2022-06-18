import spacy
import neuralcoref

from tool.coreference.coreference_model import CoreferenceModel


class NeuralCoref(CoreferenceModel):
    def __init__(self, model_name, save_singletons):
        super().__init__()
        self.model = spacy.load(model_name)
        neuralcoref.add_to_pipe(self.model)
        self.save_singletons = save_singletons


    def get_clusters(self, doc, start_cluster_id = None, save_conll = False):
        output = self.model(doc)
        if save_conll:
            return self.write_conll(output, start_cluster_id)


    def write_conll(self, output, start_cluster_id):

        results = []
        for token_id, token in enumerate(output):
            line = ['title', '0', str(token_id), str(token), '']
            results.append(line)

        cluster_id = None
        for cluster_id, cluster in enumerate(output._.coref_clusters):
            if self.save_singletons and len(cluster) < 2:
                continue
            else:
                print(cluster)
                for mention in cluster:
                    token_start = mention.start
                    token_end = mention.end - 1
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
