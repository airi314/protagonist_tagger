import argparse
import os

import tabulate

from tool.file_and_directory_management import open_path, load_from_pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', type=str)
    parser.add_argument('stats_path', type=str,
                        help="path to .txt file with titles of novels for which NER model should be tested")
    parser.add_argument('--save_path', type=str,
                        help="if True then results will be saved in chosen path")
    opt = parser.parse_args()

    headers = ['Model', 'Precision', 'Recall', 'F-measure']
    metrics_table = []

    libs = ['nltk', 'spacy__en_core_web_sm', 'spacy__en_core_web_md', 'spacy__en_core_web_lg', 'spacy__en_core_web_trf',
            'flair__ner-fast', 'flair__ner', 'flair__ner-large', 'stanza__ontonotes', 'stanza__conll03']
    for library in libs:
        try:
            results = load_from_pickle(os.path.join('experiments', opt.results_dir,
                                                    library, opt.stats_path, 'overall_metrics'))
            library = library.replace('__', ' ')
            metrics_table.append([library] + results[:3])
        except BaseException:
            pass

    results = tabulate.tabulate(
        metrics_table,
        headers=headers,
        tablefmt='latex_booktabs')
    print(results)

    if opt.save_path:
        open_path(opt.save_path, 'w').write(results)
