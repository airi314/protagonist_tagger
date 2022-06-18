import torch
import random
import numpy as np
import os
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.datasets import ColumnCorpus
import argparse
from tool.file_and_directory_management import dir_path, mkdir


def main(training_set_dir_path, output_dir, random_seed, max_epochs,
         mini_batch_size, learning_rate):
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    mkdir(output_dir)

    tagger = SequenceTagger.load('ner-large')
    columns = {0: 'text', 1: 'ner'}

    corpus = ColumnCorpus('.', columns,
                            train_file=os.path.join(training_set_dir_path, 'train.txt'),
                            dev_file=os.path.join(training_set_dir_path, 'dev.txt')
                            )

    trainer = ModelTrainer(tagger, corpus)
    trainer.train(output_dir,
                  learning_rate=learning_rate,
                  mini_batch_size=mini_batch_size,
                  save_model_each_k_epochs=5,
                  max_epochs=max_epochs,
                  train_with_test=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('training_set_dir_path', type=dir_path)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('random_seed', type=int, default=42, nargs='?')
    parser.add_argument('max_epochs', type=int, default=10, nargs='?')
    parser.add_argument('mini_batch_size', type=int, default=8, nargs='?')
    parser.add_argument('learning_rate', type=float, default=0.01, nargs='?')

    opt = parser.parse_args()
    main(opt.training_set_dir_path, opt.output_dir, opt.random_seed,
         opt.max_epochs, opt.mini_batch_size, opt.learning_rate)