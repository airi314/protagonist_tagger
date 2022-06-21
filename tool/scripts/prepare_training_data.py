import os
import json
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
import argparse

from tool.file_and_directory_management import read_file_to_list, dir_path, write_text_to_file, mkdir, read_sentences_from_file
from tool.annotations_utils import read_annotations

COMMON_NAMES_FILE = "tool/additional_resources/common_names.txt"


def main(dataset_name, output_dir, gold_standard_path):
    mkdir(output_dir)

    if dataset_name == 'litbank':
        titles = read_file_to_list('data/novels_titles/litbank.txt')
        train, val = titles[:80], titles[80:]
    else:
        titles = read_file_to_list('data/novels_titles/combined_set.txt')
        train, val = titles[:6], titles[6:9]
        model_checkpoint = "Jean-Baptiste/roberta-large-ner-english"
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        if dataset_name == 'protagonist_artificial':
            prepare_artificial_data(titles, output_dir, gold_standard_path)

    for data, file in zip([train, val], ['train.txt', 'dev.txt']):
        document = ''
        for title in data:
            if dataset_name == "litbank":
                sentences = read_sentences_from_file('resources/litbank/entities/brat/' + title.replace('tsv', 'txt'))
                data = pd.read_csv(
                    'resources/litbank/coref/tsv/' +
                    title.replace(
                        'tsv',
                        'ann'),
                    sep='\t',
                    header=None)
                data.columns = [
                    'type',
                    'id',
                    'sent',
                    'token_start',
                    'sent2',
                    'token_end',
                    'text',
                    'tag',
                    'tag_type']
                data = data[data.type == 'MENTION']
                data = data[data.tag_type == 'PROP']
                data = data[(data.tag == 'LOC') | (data.tag == 'PER')]

                for sent_id, sent in enumerate(sentences):
                    document_sent = ''
                    tokens = sent.split()
                    entities = data.loc[data.sent2 == sent_id, [
                        'token_start', 'token_end', 'tag']].values
                    entities = sorted(entities, key=lambda x: x[0])
                    entity_id = 0
                    token_id = 0
                    while token_id < len(tokens):
                        if entity_id == len(
                                entities) or entities[entity_id][0] > token_id:
                            document_sent += tokens[token_id] + \
                                '\t' + 'O' + '\n '
                            token_id += 1
                        elif entities[entity_id][0] == token_id:
                            document_sent += tokens[int(entities[entity_id][0])] + \
                                '\t' + 'B-' + entities[entity_id][2] + '\n'
                            for t in range(
                                    int(entities[entity_id][0]) + 1, int(entities[entity_id][1] + 1)):
                                document_sent += tokens[t] + '\t' + \
                                    'I-' + entities[entity_id][2] + '\n'
                            token_id += int(entities[entity_id]
                                            [1] - entities[entity_id][0] + 1)
                            entity_id += 1
                        elif entity_id != len(entities) and entities[entity_id][0] < token_id:
                            entity_id += 1
                    document += document_sent + '\n'
            else:
                if dataset_name == 'protagonist':
                    annotations = read_annotations(
                        os.path.join(gold_standard_path, title + '.json'))
                else:
                    annotations = read_annotations(
                        os.path.join(output_dir, title + '.json'))

                for sent in annotations:
                    tokens = []
                    starts = []
                    for w in range(tokenizer(sent['content']).word_ids()[-2]):
                        charspan = tokenizer(sent['content']).word_to_chars(w)
                        tokens.append(
                            sent['content'][charspan.start:charspan.end])
                        starts.append(charspan.start)
                    results = []
                    ent_id = 0
                    i = 0
                    entities = sorted(sent['entities'], key=lambda x: x[0])
                    while ent_id < len(entities) and i < len(starts):
                        if starts[i] < entities[ent_id][0]:
                            results.append('O')
                            i += 1
                        elif starts[i] == entities[ent_id][0]:
                            results.append('B-PER')
                            i += 1
                        elif starts[i] < entities[ent_id][1]:
                            results.append('I-PER')
                            i += 1
                        else:
                            ent_id += 1
                    while i < len(starts):
                        results.append('O')
                        i += 1
                    document += '\n'.join([t + '\t' + r for t, r in
                                           zip(tokens, results)]) + '\n\n'

        write_text_to_file(os.path.join(output_dir, file), document)


def prepare_artificial_data(titles, output_dir, gold_standard_path):
    common_names = read_file_to_list(COMMON_NAMES_FILE)
    for title in titles:
        annotations = read_annotations(
            os.path.join(gold_standard_path, title + '.json'))
        for anno in annotations:
            if anno['entities']:
                ent_id = np.random.choice(np.arange(len(anno['entities'])))
                ent = anno['entities'][ent_id]
                ent_text = anno['content'][ent[0]:ent[1]]
                ent_text = ent_text.split(' ')[0]
                name = np.random.choice(common_names)
                anno['content'] = anno['content'].replace(ent_text, name)
                ent[1] += len(name) - len(ent_text)
        write_text_to_file(os.path.join(output_dir, title + '.json'), json.dumps(annotations))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str,
                        help='Name of dataset. One of protagonist, protagonist_artificial or litbank.')
    parser.add_argument('output_dir', type=str,
                        help="directory where generated data should be stored")
    parser.add_argument('gold_standard_path', type=dir_path,
                        help="directory where gold-standard data are stored")
    opt = parser.parse_args()
    main(opt.dataset_name, opt.output_dir, opt.gold_standard_path)
