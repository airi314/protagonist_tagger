import argparse
import os
import json
from tqdm import tqdm
from collections import Counter
import numpy as np

from tool.file_and_directory_management import read_file_to_list, file_path, \
    dir_path, mkdir, write_text_to_file
from tool.annotations_utils import read_annotations


def main(titles_path, protagonist_data_dir, coreference_data_dir,
         generated_data_dir):
    titles = read_file_to_list(titles_path)
    mkdir(generated_data_dir)

    for title in tqdm(titles):
        combined_results = []
        protagonist_results = read_annotations(os.path.join(
            protagonist_data_dir, title + '.json'))[0]
        ner_results = read_annotations(os.path.join(
            protagonist_data_dir, 'ner', title + '.json'))[0]
        coref_results = read_annotations(
            os.path.join(coreference_data_dir, title + '.json'))

        matched_clusters = 0
        matched_entities = [False] * len(ner_results['entities'])
        for cluster in coref_results['clusters']:
            possible_matches = []
            for ent_id, (ent_ner, ent_protagonist) in enumerate(
                    zip(ner_results['entities'],
                        protagonist_results['entities'])):
                ent_personal = None
                if ent_ner[3]:
                    ent_personal = [ent_protagonist[0] - len(ent_ner[3]) - 1,
                                    ent_protagonist[1], ent_protagonist[2]]
                if ent_protagonist[:2] in cluster or (
                        ent_personal and ent_personal[:2] in cluster):
                    possible_matches.append(ent_protagonist[2])
                    matched_entities[ent_id] = True
            match = None
            if len(possible_matches) > 1:
                counts = Counter(possible_matches).most_common()
                matches = [counts[0][0]]
                max_count = counts[0][1]
                i = 0
                while i < len(counts) and counts[i][1] == max_count:
                    matches.append(counts[i][0])
                    i += 1
                match = np.random.choice(matches)

            elif len(possible_matches) == 1:
                match = possible_matches[0]

            if match:
                matched_clusters += 1
                for mention in cluster:
                    combined_results.append((mention[0], mention[1], match))

        for ent_id, ent in enumerate(protagonist_results['entities']):
            if not matched_entities[ent_id] and ent not in combined_results:
                ent[0] = int(ent[0])
                ent[1] = int(ent[1])
                combined_results.append(tuple(ent))

        results_dict = [{
            'content': ner_results['content'],
            'mentions': combined_results}]
        path = os.path.join(generated_data_dir, title + ".json")
        write_text_to_file(path, json.dumps(results_dict))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('titles_path', type=file_path,
                        help="path to .txt file with titles of novels for which NER model should be tested")
    parser.add_argument('protagonist_data_dir', type=dir_path,
                        help="path to the directory with results of protagonist matching")
    parser.add_argument('coreference_data_dir', type=dir_path,
                        help="path to the directory with results of coreference resolution")
    parser.add_argument('generated_data_dir', type=str,
                        help="directory where generated data should be stored")
    opt = parser.parse_args()
    main(
        opt.titles_path,
        opt.protagonist_data_dir,
        opt.coreference_data_dir,
        opt.generated_data_dir)