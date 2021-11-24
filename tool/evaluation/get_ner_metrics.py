import os
import tabulate
import argparse

from tool.file_and_directory_management import load_from_pickle, file_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', type=str)
    parser.add_argument('stats_path', type=str,
                        help="path to .txt file with titles of novels for which NER model should be tested")
    parser.add_argument('--protagonist_tagger', action='store_true',
                        help="if metrics for protagonist_tagger should be printed")
    parser.add_argument('--save_path', type=str,
                        help="if True then results will be saved in chosen path")
    opt = parser.parse_args()

    headers = ['Model', 'Precision', 'Recall', 'F-measure']
    metrics_table = []

    if opt.protagonist_tagger:
        task = 'protagonist'
    else:
        task = 'ner'

    for library in sorted(os.listdir(os.path.join('experiments', opt.results_dir, task))):
        results = load_from_pickle(os.path.join('experiments', opt.results_dir,
                                                task, library, opt.stats_path, 'overall_metrics'))
        library = library.replace('__', ' ')
        metrics_table.append([library] + results[:3])
    print(tabulate.tabulate(metrics_table, headers=headers, tablefmt='latex_booktabs'))

    if opt.save_path:
        with open(opt.save_path, 'w') as f:
            f.write(tabulate.tabulate(metrics_table, headers=headers, tablefmt='latex_booktabs'))
