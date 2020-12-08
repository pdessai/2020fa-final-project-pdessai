import os
import string

import luigi
from luigi import ExternalTask, Parameter, format, Task, LocalTarget

from csci_utils.luigi.target import SuffixPreservingLocalTarget
from country_name_recognize.rnn import *


class InputModel(ExternalTask):
    MODEL_ROOT = os.path.abspath('data')
    model = Parameter(default="rnn.pth")  # Filename of the model

    def output(self):
        return SuffixPreservingLocalTarget(self.MODEL_ROOT + '/' + 'models' + '/' + self.model, format=format.Nop)


class InputData(ExternalTask):
    IMAGE_ROOT = os.path.abspath('data')
    data = Parameter(default="names.txt")  # Filename of the model

    def output(self):
        return SuffixPreservingLocalTarget(self.IMAGE_ROOT + '/' + 'input' + '/' + self.data, format=format.Nop)


class Predict_names(Task):
    model = Parameter(default="rnn.pth")
    data = Parameter(default="names.txt")
    LOCAL_ROOT = os.path.abspath('data')
    SHARED_RELATIVE_PATH = 'output'

    outputs = Parameter(default="output.csv")  # Luigi parameter

    def requires(self):
        data_path = self.LOCAL_ROOT + '/' + 'input' + '/' + self.data
        model_path = self.LOCAL_ROOT + '/' + 'models' + '/' + self.model
        return {
            'data': self.clone(InputData),
            'model': self.clone(InputModel)
        }

    def output(self):
        # return SuffixPreservingLocalTarget of the predicted country names
        output_path = os.path.join(self.LOCAL_ROOT, self.SHARED_RELATIVE_PATH, str(self.outputs))
        return SuffixPreservingLocalTarget(output_path, format=format.Nop)

    def run(self):
        # For example
        inputs = self.input()
        category_lines = {}
        all_categories = []
        all_letters = string.ascii_letters + " .,;'"
        n_letters = len(all_letters)

        for filename in find_files('data/names/*.txt'):
            category = os.path.splitext(os.path.basename(filename))[0]
            all_categories.append(category)
            lines = read_lines(filename)
            category_lines[category] = lines
        n_categories = len(all_categories)

        with self.output().temporary_path() as temp_output_path:
            class args:
                with open(self.input()['data'].path, 'r') as f:
                #with self.input()['data'].open() as f:
                    input_lines = f.read()
                print(input_lines)
                output_path = temp_output_path
                model = inputs['model'].path
                n_predictions = 3

            predict_country_name(args,all_categories, n_letters)
