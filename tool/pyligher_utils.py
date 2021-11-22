import os
import json
import pandas as pd
import ast
from pylighter import Annotation

from tool.gender_checker import get_personal_titles


def read_annotations(annotation_path):
    with open(annotation_path) as f:
        annotations = f.read()
    annotations = json.loads(annotations)
    return annotations


def annotations_to_pylighter(annotations):
    labels = []
    corpus = []
    for sentence in annotations:
        text = sentence['content']
        corpus.append(text)
        entities = sentence['entities']

        sent_labels = ['O']*len(text)
        for ent in entities:
            sent_labels[ent[0]] = 'B-' + ent[2]
            for index in range(ent[0]+1, ent[1]):
                sent_labels[index] = 'I-' + ent[2]
        labels.append(sent_labels)
    return labels, corpus


def fix_personal_titles(annotations):
    personal_titles = tuple(get_personal_titles)
    for anno in annotations:
        for ent in anno['entities']:
            text = anno['content'][ent[0]:ent[1]]
            if text.startswith(personal_titles):
                ent[0] += (1 + len(text.split(' ')[0]))
    return annotations


def personal_titles_stats(annotations):
    personal_title_annotated = {}
    personal_title_not_annotated = {}

    for title in PERSONAL_TITLES:
        personal_title_annotated[title] = 0
        personal_title_not_annotated[title] = 0

    for anno in annotations:
        for ent in anno['entities']:
            text = anno['content'][ent[0]:ent[1]]
            if text.startswith(PERSONAL_TITLES):
                title = text.split(' ')[0]
                personal_title_annotated[title] += 1
            elif any(ext in anno['content'][(ent[0] - 8):ent[0]] for ext in PERSONAL_TITLES):
                title = anno['content'][(ent[0] - 8):ent[0]].split(' ')[-2].strip('"').strip("'")
                personal_title_not_annotated[title] += 1

    return personal_title_annotated, personal_title_not_annotated


def csv_to_json(csv_path, json_path):
    df = pd.read_csv(csv_path, sep=';')
    new_annotations = []
    for sent_id, (text, labels) in enumerate(zip(df.document, df.labels)):
        sent_entities = []
        started = False
        for i, tag in enumerate(ast.literal_eval(labels)):
            if tag.startswith('B'):
                started = True
                entity_type = tag[2:]
                start = i
            if started and tag == 'O':
                started = False
                end = i
                sent_entities.append([start, end, entity_type])
        if started:
            sent_entities.append([start, i+1, entity_type])
        new_annotations.append({'content': text, 'entities': sent_entities})
    with open(json_path, 'w') as f:
        f.write(json.dumps(new_annotations))


def has_intersection(ent_1, ent_2):
    if ent_1[1] <= ent_2[0] or ent_2[1] <= ent_1[0]:
        return False
    return True