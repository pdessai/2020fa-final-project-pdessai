import os
import unittest

from luigi import format, build
from csci_utils.luigi.target import SuffixPreservingLocalTarget
from rnn_tasks.tasks.rnn import Predict_names, InputData, InputModel


class PredictNamesTests(unittest.TestCase, Predict_names, InputData, InputModel):
    def test_predict_name_file_exits(self):
        path = 'data/output/output.csv'
        def main():
            build([
                Predict_names(
                    data='names.txt',
                    model='rnn.pth',
                    outputs='output.csv'
                )], local_scheduler=True)

        main()
        self.assertTrue(os.path.exists(path))


    def test_InputData_exits(self):
        path = 'data/input/names.txt'

        def main():
            build([
                InputData(
                    data='names.txt'
                )], local_scheduler=True)

        main()
        self.assertTrue(os.path.exists(path))