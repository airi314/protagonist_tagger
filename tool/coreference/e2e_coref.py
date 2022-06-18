import os
import sys
sys.path.append('resources/coref-hoi/')

from tool.coreference.coreference_model import CoreferenceModel
from run import Runner
from tensorize import CorefDataProcessor
from predict import create_spacy_tokenizer, get_document_from_string
from spacy.lang.en import English


class E2ECoref(CoreferenceModel):
    def __init__(self, model_name, save_singletons):
        super().__init__()
        model_name, model_identifier = model_name.split('/')
        os.chdir('resources/coref-hoi/')
        self.runner = Runner(model_name, None)
        self.data_processor = CorefDataProcessor(self.runner.config)
        model = self.runner.initialize_model(model_identifier)
        model.to(model.device)
        self.model = model
        self.spacy_tokenizer = English()
        self.spacy_tokenizer.add_pipe(self.spacy_tokenizer.create_pipe('sentencizer'))
        self.save_singletons = save_singletons
        os.chdir('../..')

    def get_clusters(self, doc, start_cluster_id = None, save_conll = False):
        doc = get_document_from_string(doc, 512, self.data_processor.tokenizer, self.spacy_tokenizer)
        tensor_examples, stored_info = self.data_processor.get_tensor_examples_from_custom_input([doc])
        clusters, spans, antecedents = self.runner.predict(self.model, tensor_examples)

        doc['clusters'] = clusters
        if save_conll:
            lines, start_cluster_id = self.write_conll(doc, start_cluster_id)
            return lines, start_cluster_id

    def write_conll(self, output, start_cluster_id):
        results = []
        for token_id, token in enumerate(output['tokens']):
            line = ['title', '0', str(token_id), token, '']
            results.append(line)

        cluster_id = None
        if output['clusters'][0]:
            for cluster_id, cluster in enumerate(output['clusters'][0]):
                if not self.save_singletons and len(cluster) < 2:
                    continue
                else:
                    for mention in cluster:
                        token_start = output['subtoken_map'][mention[0]]
                        token_end = output['subtoken_map'][mention[1]]
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
