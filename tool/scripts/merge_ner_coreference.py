import argparse
import os
import json
from tqdm import tqdm
from collections import Counter
import numpy as np

from tool.file_and_directory_management import read_file_to_list, file_path


def main(titles_path, protagonist_data_dir, coreference_data_dir,
         generated_data_dir, skip_ambiguous):
    titles = read_file_to_list(titles_path)

    for title in tqdm(titles):
        combined_results = []
        with open(os.path.join(protagonist_data_dir, title + '.json')) as f:
            protagonist_results = json.loads(f.read())

        with open(os.path.join(protagonist_data_dir, 'ner', title + '.json')) as f:
            ner_results = json.loads(f.read())

        with open(os.path.join(coreference_data_dir, title + '.json')) as f:
            coref_results = json.loads(f.read())

        for cluster in coref_results['clusters']:
            possible_matches = []
            matches = []
            for ent_ner, ent_protagonist in zip(
                    ner_results[0]['entities'], protagonist_results[0]['entities']):
                ent_ner[0] -= len(ent_ner[3]) + 1
                if ent_protagonist[:2] in cluster or ent_ner[:2] in cluster:
                    matches.append(ent_protagonist)
                    possible_matches.append(ent_protagonist[:2])
            if len(possible_matches) > 1:
                if not skip_ambiguous:
                    counts = Counter(possible_matches).most_common()
                    matches = [counts[0][0]]
                    max_count = counts[0][1]
                    i = 1
                    while counts[i][1] == max_count:
                        matches.append(counts[i][0])
                        i += 1
                    match = np.random.choice(matches)
            else:
                matches = possible_matches[0]
            for mention in cluster:
                combined_results.append((mention[0], mention[1], match))

        results_dict = {
            'content': ner_results[0]['content'],
            'mentions': combined_results}
        path = os.path.join(generated_data_dir, title + ".json")

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        with open(path, 'w+') as result:
            json.dump(results_dict, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('titles_path', type=file_path,
                        help="path to .txt file with titles of novels for which NER model should be tested")
    parser.add_argument('protagonist_data_dir', type=str)
    parser.add_argument('coreference_data_dir', type=str)
    parser.add_argument('generated_data_dir', type=str,
                        help="directory where generated data should be stored")
    parser.add_argument('--skip_ambiguous', action='store_true')
    opt = parser.parse_args()
    main(
        opt.titles_path,
        opt.ner_data_dir,
        opt.coreference_data_dir,
        opt.generated_data_dir,
        opt.skip_ambiguous)
