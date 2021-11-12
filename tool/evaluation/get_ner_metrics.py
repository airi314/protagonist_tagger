import os
import tabulate
from tool.ner_metrics import load_from_pickle


headers = ['Model', 'Precision', 'Recall', 'F-measure']

for subset in ['small_set', 'large_set']:
    print('Results table for ' + subset)
    metrics_table = []
    for library in sorted(os.listdir('experiments/ner')):
        results = load_from_pickle('overall_metrics', os.path.join('experiments/ner', library, subset, 'stats'))
        library = library.replace('__', ' ')
        metrics_table.append([library] + results[:3])
    print(tabulate.tabulate(metrics_table, headers=headers, tablefmt='latex_booktabs'))
