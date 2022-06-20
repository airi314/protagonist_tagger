import os
import re

from tool.file_and_directory_management import read_file, read_file_to_list, \
    read_sentences_from_file


def get_litbank_sentences(path):
    text = read_file(path)
    sentences = text.split('\n')
    return sentences


def get_litbank_text(path):
    sentences = get_litbank_sentences(path)
    text = ' '.join(sentences)
    return text


def get_litbank_parts(path, max_length=512):
    sentences = read_sentences_from_file(path)
    parts = []
    current_part = ''
    counter = 0
    for sent in sentences:
        n = len(sent.split(' '))
        if counter + n <= max_length:
            current_part += ' ' + sent
            counter += n
        else:
            parts.append(current_part)
            current_part = sent
            counter = n
    return parts


def get_test_data_for_novel(title, testing_data_dir_path,
                            gutenberg, pride_and_prejudice):
    if pride_and_prejudice:
        return get_pride_and_prejudice(title, testing_data_dir_path)
    elif gutenberg:
        return get_gutenberg(title, testing_data_dir_path, 10000)
    else:
        return read_sentences_from_file(
            os.path.join(testing_data_dir_path, title))


def get_pride_and_prejudice(title, testing_data_dir_path):
    text = read_file(os.path.join(testing_data_dir_path, title))
    return text.split('\n\n')


def get_gutenberg(title, testing_data_dir_path, max_length):
    text = read_file(os.path.join(testing_data_dir_path, title))
    start = re.search("\*\*\* START OF .*", text).span()[1]
    end = re.search("\*\*\* END OF .*", text).span()[0]
    text = text[start:end].strip()
    if not max_length:
        return [text]
    text_parts = text.split('\n\n')
    text_parts = [text_part.strip().replace('\n', ' ')
                  for text_part in text_parts]
    parts = []
    current_part = ''
    counter = 0
    for part in text_parts:
        n = len(part)
        if counter + n <= max_length:
            current_part += ' ' + part
            counter += n
        else:
            parts.append(current_part)
            current_part = part
            counter = n
    return parts


def get_characters_for_novel(title, characters_lists_dir_path):
    return read_file_to_list(os.path.join(characters_lists_dir_path, title))
