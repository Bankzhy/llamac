import csv
import math
import os
import sys
import random
import re
from torch.utils.data.dataset import Dataset

class SREDataset(Dataset):

    def __init__(self, folder_path):
        """
        :param path:dataset dir path
        :param task: [mass(Masked Span Prediction), nsp(Natural Language Prediction)]
        :param language:
        """
        super(SREDataset, self).__init__()
        self.examples=self.load_examples(folder_path)


    def __len__(self):
        return len(self.texts)


    def __getitem__(self, index):

        return self.texts[index], self.fix_texts[index]


    def load_examples(self, path):
        folder_path = path

        csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
        examples = []
        for file in csv_files:
            with open(os.path.join(folder_path, file), 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=';')
                print(f"Reading {file}:")
                for row in csv_reader:
                    example = {}
                    row_list = row  # ace with your processing logic
                    example["instruction"] = row_list[0]
                    example["input"] = ""
                    example["output"] = row_list[1]
                    examples.append(example)
                    print(row_list)
                print()  # Add an empty line for separation
        return examples