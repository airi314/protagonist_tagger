import os
import tabulate
from tool.file_and_directory_management import load_from_pickle


headers = ['Model', 'Precision', 'Recall', 'F-measure']

for subset in ['small_set', 'large_set']:
    metrics_table = []
    for library in sorted(os.listdir('experiments/protagonist')):
        results = load_from_pickle(os.path.join('experiments/protagonist', library, subset, 'stats', 'overall_metrics'))
        library = library.replace('__', ' ')
        metrics_table.append([library] + results[:3])
    print(tabulate.tabulate(metrics_table, headers=headers, tablefmt='latex'))
