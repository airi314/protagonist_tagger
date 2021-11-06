import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import tabulate
import os

from tool.file_and_directory_management import read_file_to_list, load_from_pickle, save_to_pickle
from tool.data_generator import data_from_json


def organize_entities(entities_gold, entities_matcher):
    gold = []
    matcher = []

    for sent_id, sent_gold_entities in enumerate(entities_gold):
        sent_gold = []
        sent_matcher = []
        sent_matcher_entities = entities_matcher[sent_id]

        for gold_entity in sent_gold_entities:
            sent_gold.append(gold_entity[2])
            if gold_entity in sent_matcher_entities:
                # print(gold_entity, "TP")
                sent_matcher.append(gold_entity[2])
            else:
                # print(gold_entity, "FN")
                sent_matcher.append('-')

        for matcher_entity in sent_matcher_entities:
            if matcher_entity not in sent_gold_entities:
                # print(matcher_entity, 'FP')
                sent_gold.append('-')
                sent_matcher.append(matcher_entity[2])

        gold.extend(sent_gold)
        matcher.extend(sent_matcher)

    return gold, matcher


def calculate_metrics(gold, matcher, ner=True):
    characters = list(set(gold + matcher))
    if '-' in characters:
        characters.remove('-')

    if ner:
        result = precision_recall_fscore_support(
            np.array(gold),
            np.array(matcher),
            labels=characters)
    else:
        result = precision_recall_fscore_support(
            np.array(gold),
            np.array(matcher),
            labels=characters,
            average='weighted')

    result = list(result)
    result[0:3] = [np.round(a, 3) for a in result[0:3]]

    return result


def create_overall_stats(titles, gold_standard_path,
                         result_path, stats_path, ner=False):
    gold_overall = []
    matcher_overall = []

    for title in titles:
        entities_gold, _ = data_from_json(
            os.path.join(gold_standard_path, title + '.json'))
        entities_matcher, _ = data_from_json(
            os.path.join(result_path, title + '.json'))

        entities_gold = [[list(x) for x in set(tuple(x) for x in sent_gold_entities)] for sent_gold_entities in
                         entities_gold]
        gold, matcher = organize_entities(entities_gold, entities_matcher)
        gold_overall.extend(gold)
        matcher_overall.extend(matcher)
        metrics = calculate_metrics(gold, matcher, ner)
        save_to_pickle(metrics, os.path.join(stats_path, title))

    metrics = calculate_metrics(gold_overall, matcher_overall, ner)
    save_to_pickle(metrics, os.path.join(stats_path, "overall_metrics"))
    return metrics


def characters_tags_metrics(
        titles_path, gold_standard_path, result_path, stats_path, print_results):
    titles = read_file_to_list(titles_path)

    if not os.path.exists(stats_path):
        os.makedirs(stats_path)

    metrics_overall = create_overall_stats(
        titles,
        gold_standard_path,
        result_path,
        stats_path,
        ner=False)

    if print_results:
        metrics_table = []
        headers = ["Novel title", "precision", "recall", "F-measure"]

        for title in titles:
            metrics = load_from_pickle(os.path.join(stats_path, title))
            metrics_title = [title].__add__([m for m in metrics])
            metrics_table.append(metrics_title)

        metrics_table.append(
            ["*** overall results ***"].__add__([m for m in metrics_overall]))
        print(tabulate.tabulate(metrics_table, headers=headers, tablefmt='latex'))


def ner_metrics(titles_path, gold_standard_path, result_path, stats_path, print_results):
    titles = read_file_to_list(titles_path)

    if not os.path.exists(stats_path):
        os.makedirs(stats_path)

    metrics_overall = create_overall_stats(
        titles,
        gold_standard_path,
        result_path,
        stats_path,
        ner=True)

    if print_results:
        metrics_table = []
        headers = ["Novel title", "precision", "recall", "F-measure", "support"]

        for title in titles:
            metrics = load_from_pickle(os.path.join(stats_path, title))
            metrics_title = [title].__add__([m[0] for m in metrics])
            metrics_table.append(metrics_title)

        metrics_table.append(
            ["*** overall results ***"].__add__([m[0] for m in metrics_overall]))
        print(tabulate.tabulate(metrics_table, headers=headers, tablefmt='latex'))


def ner_metrics_on_files(gold_standard_file, results_file, ner = True):
    entities_gold, _ = data_from_json(gold_standard_file)
    entities_matcher, _ = data_from_json(results_file)

    entities_gold = [[list(x) for x in set(tuple(x) for x in sent_gold_entities)] for sent_gold_entities in
                     entities_gold]
    gold, matcher = organize_entities(entities_gold, entities_matcher)
    metrics = calculate_metrics(gold, matcher, ner)
    print(metrics)
