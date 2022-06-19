import os

from tool.file_and_directory_management import read_file, read_file_to_list, read_sentences_from_file


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


def get_test_data_for_novel(title, testing_data_dir_path, full_text):
    if full_text:
        return get_litbank_text(os.path.join(testing_data_dir_path, title))
    else:
        return read_sentences_from_file(
            os.path.join(testing_data_dir_path, title))


def get_pride_and_prejudice(title, testing_data_dir_path, full_text=True):
    text = read_file(os.path.join(testing_data_dir_path, title))
    if full_text:
        return text
    else:
        return text.split('\n\n')
    # print(parts[0])
    # parts = [' '.join(p.split('\n')) for p in parts]
    # print(parts[0])
    # if full_text:
    #     return '\n\n'.join(parts)


def get_characters_for_novel(title, characters_lists_dir_path):
    return read_file_to_list(os.path.join(characters_lists_dir_path, title))
