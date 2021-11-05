import pickle
import os
import tabulate


def load_from_pickle(filename, stats_path):
    file = open(os.path.join(stats_path, filename), "rb")
    data = pickle.load(file)
    return data


headers = ['Model', 'Precision', 'Recall', 'F-measure']

for subset in ['small_set', 'large_set']:
    metrics_table = []
    for library in sorted(os.listdir('experiments/protagonist')):
        results = load_from_pickle('overall_metrics', os.path.join('experiments/protagonist', library, subset, 'stats'))
        library = library.replace('__', ' ')
        metrics_table.append([library] + results)
    print(tabulate.tabulate(metrics_table, headers=headers, tablefmt='latex'))
