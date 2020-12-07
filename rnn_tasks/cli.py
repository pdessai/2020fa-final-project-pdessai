import argparse
from luigi import build

from .tasks.rnn import Predict_names


parser = argparse.ArgumentParser(description='Command description.')

def main(args=None):
    args = parser.parse_args(args=args)

    build([
        Predict_names(
            data='names.txt',
            model='rnn_lstm.pkl',
            outputs='output.csv'
        )], local_scheduler=True)