import tabulate
import os
import subprocess
import pandas as pd
import re
import numpy as np
from tqdm import tqdm

from tool.file_and_directory_management import read_file_to_list, \
    save_to_pickle, load_from_pickle, write_text_to_file


def load_conll(filepath, litbank=False):
    with open(filepath) as f:
        data = f.read()
    if litbank:
        data = pd.DataFrame([d.split('\t') for d in data.split('\n')[1:-2]])
        data.columns = ['title', 'id', 'token_ID', 'text'] + \
                       ['col' + str(i) for i in range(4, 12)] + ['mention']
        data = data[~data.id.isna()].sort_index().reset_index(drop=True)
        data = data[['title', 'id', 'token_ID', 'text', 'mention']]
    else:
        data = pd.DataFrame([d.split('\t') for d in data.split('\n')])
        data.columns = ['title', 'id', 'token_ID', 'text', 'mention']
    return data


def organize_entities(gold, pred):
    i = 0
    quotes = ['“', '”', '"', '’”', '“‘', '—', '’', '‘']
    while i < gold.shape[0] or i < pred.shape[0]:
        if i < pred.shape[0]:

            t2 = pred.iloc[i].text
            if i == gold.shape[0] and t2.strip() == '':
                pred = pred.drop(i).reset_index(drop=True)
                continue
            t1 = gold.iloc[i].text
            if t1 != t2:
                if t1 in quotes and t2 and t2[0] not in quotes:
                    line = pd.DataFrame(dict(pred.iloc[i]), index=[i - 0.5])
                    line.text = t1
                    line.mention = '-'
                    pred = pd.concat([pred, line]).sort_index().reset_index(
                        drop=True)
                elif t2.strip() == t1.replace('’',
                                              '') or t2.strip() == t1.replace(
                        '‘', ''):
                    pred.iloc[i].text = t1
                elif t2.strip() == '' or ((t2 == '-') & (t1[0] != '-')) or (
                        (t2 in quotes) & (t1[0] not in quotes)):
                    if pred.iloc[i].mention != '-':
                        pred.iloc[i - 2].mention += pred.iloc[i].mention
                    pred = pred.drop(i).reset_index(drop=True)
                    continue
                elif len(t1) > len(t2):
                    if t1.encode('ascii', 'ignore').decode() == t2:
                        pred.iloc[i].text = t1
                    else:
                        line = pd.DataFrame(dict(gold.iloc[i]),
                                            index=[i - 0.5])
                        line.iloc[0].text = t2
                        mention = line.iloc[0].mention
                        if mention:
                            mentions = re.findall('[(]?\\d+[)]?', mention)
                            line.mention = ''
                            gold.iloc[i].mention = ''
                            for m in mentions:
                                if m[0] == '(' and m[-1] == ')':
                                    line.mention += m[:-1]
                                    gold.iloc[i].mention += m[1:]
                                elif m[0] == '(':
                                    line.mention += m
                                else:
                                    gold.iloc[i].mention += m
                        gold.iloc[i].text = gold.iloc[i].text[len(t2):]
                        gold = pd.concat(
                            [gold, line]).sort_index().reset_index(drop=True)
                else:
                    line = pd.DataFrame(dict(pred.iloc[i]), index=[i - 0.5])
                    line.text = t1
                    mention = line.iloc[0].mention
                    if mention:
                        mentions = re.findall('[(]?\\d+[)]?', mention)
                        line.mention = ''
                        pred.iloc[i].mention = ''
                        for m in mentions:
                            if m[0] == '(' and m[-1] == ')':
                                line.mention += m[:-1]
                                pred.iloc[i].mention += m[1:]
                            elif m[0] == '(':
                                line.mention += m
                            else:
                                pred.iloc[i].mention += m
                    pred.iloc[i].text = pred.iloc[i].text[len(t1):]
                    pred = pd.concat([pred, line]).sort_index().reset_index(
                        drop=True)
        else:
            t2 = gold.iloc[i].text
            line = pd.DataFrame(
                {'title': 'title', 'token_ID': i, 'text': t2, 'mention': '-'},
                index=[i])
            pred = pd.concat([pred, line]).sort_index().reset_index(drop=True)
        i += 1

    gold.id = 0
    pred.id = 0
    gold.token_ID = np.arange(gold.shape[0])
    pred.token_ID = np.arange(pred.shape[0])
    assert np.all(pred.text == gold.text)

    gold.loc[gold.mention == '', 'mention'] = '-'
    gold.loc[:, 'mention'] = [''.join(sorted(mention.split('|'), reverse=True))
                              for mention in gold.mention]
    pred.loc[:, 'mention'] = [''.join(sorted(mention.split('|'), reverse=True))
                              for mention in pred.mention]

    return gold, pred


def convert_files(gold, pred):
    pred_file = '#begin document (Test);\n'
    gold_file = '#begin document (Test);\n'
    pred_file += pred.to_csv(sep='\t', header=None, index=False) + '\n'
    gold_file += gold.to_csv(sep='\t', header=None, index=False) + '\n'
    pred_file += '#end document'
    gold_file += '#end document'
    return gold_file, pred_file


def calculate_metrics(gold, pred):
    gold_file, pred_file = convert_files(gold, pred)
    write_text_to_file('/tmp/gold.tsv', gold_file)
    write_text_to_file('/tmp/pred.tsv', pred_file)

    results = subprocess.run(
        ['perl',
         'resources/reference-coreference-scorers/scorer.pl',
         'all',
         '/tmp/gold.tsv',
         '/tmp/pred.tsv',
         'none'],
        stdout=subprocess.PIPE)

    metrics_parsed = results.stdout.decode('utf-8').split('METRIC')[1:]
    metrics_parsed = [[l for l in metric.split('\n') if l.strip()] for metric
                      in
                      metrics_parsed]
    scores = [re.findall('\\d*[.]*\\d+(?=[%])', metric[-2]) for metric in
              metrics_parsed]
    metric_names = [metric[0].replace(':', '').strip() for metric in
                    metrics_parsed]
    score_metric = {}
    for score, metric_name in zip(scores, metric_names):
        score_metric[metric_name] = [float(s) for s in score]
    return score_metric


def compute_overall_stats(titles, gold_standard_path,
                          prediction_path, stats_path):
    metrics_overall = []
    for title in tqdm(titles):
        mentions_gold = load_conll(
            os.path.join(gold_standard_path, title + '_brat.conll'),
            litbank=True)
        mentions_pred = load_conll(
            os.path.join(prediction_path, title + '.conll'))

        gold, pred = organize_entities(mentions_gold, mentions_pred)
        metrics_title = calculate_metrics(gold, pred)
        metrics_overall.append(metrics_title)
        save_to_pickle(metrics_title, os.path.join(stats_path, title))

    overall_mean = {}
    for key in metrics_overall[0].keys():
        overall_mean[key] = [
            sum(d[key][i] for d in metrics_overall) / len(metrics_overall) for i in range(3)]
    save_to_pickle(overall_mean, os.path.join(stats_path, 'overall_metrics'))
    return metrics_overall


def metrics(titles_path, gold_standard_path, prediction_path, stats_path,
            print_results=False):
    titles = read_file_to_list(titles_path)

    compute_overall_stats(
        titles,
        gold_standard_path,
        prediction_path,
        stats_path
    )

    if print_results:
        results = get_results(stats_path, titles)
        print(results)


def get_results(stats_path, titles):
    metrics_table = []
    headers = ["Novel title", "MUC", "B3", "CEAFm", "CEAFe", "BLANC"]

    for title in titles:
        metrics_title = load_from_pickle(os.path.join(stats_path, title))
        metrics_table.append([title].__add__([m[2]
                             for m in list(metrics_title.values())]))

    metrics_overall = load_from_pickle(
        os.path.join(stats_path, 'overall_metrics'))
    metrics_table.append(
        ["*** overall results ***"].__add__([m[2] for m in list(metrics_overall.values())]))
    return tabulate.tabulate(metrics_table, headers=headers, tablefmt='latex')
