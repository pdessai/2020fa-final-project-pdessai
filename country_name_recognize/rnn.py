import argparse
import os
import sys
import time
import re
import math

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

from utils import *


def train_batch(args):
    # Build the category_lines dictionary, a list of names per language
    category_lines = {}
    all_categories = []
    all_letters = string.ascii_letters + " .,;'"
    n_letters = len(all_letters)
    file_path = args.train_file_path ## be an arg

    for filename in find_files('data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = read_lines(filename)
        category_lines[category] = lines

    n_categories = len(all_categories)
    learning_rate = args.learning_rate ## be an arg
    criterion = nn.CrossEntropyLoss()  ## be an arg
    n_hidden = args.n_hidden #128 ## be an arg
    n_iters = args.n_iters # 5000 be an arg
    print_every = 500
    plot_every = 100

    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []

    rnn = RNN(n_letters, n_hidden, n_categories)

    start = time.time()

    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = sample_trainning(n_letters, all_categories,category_lines)
        output, loss = train(rnn, criterion, learning_rate, category_tensor, line_tensor)
        current_loss += loss

        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            guess, guess_i = category_from_output(all_categories,output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (
            iter, iter / n_iters * 100, time_since(start), loss, line, guess, correct))

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    save_model_path = args.save_model_path
    torch.save(rnn, save_model_path)

    return all_losses, output


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def predict_country_name(args, all_categories, n_letters):
    rnn = torch.load(args.model)
    input_lines = args.input_lines # be an arg
    n_predictions = args.n_predictions # 3, be an arg

    res = {}

    for input_line in input_lines.split('_'):
        print('\n> %s' % input_line)
        with torch.no_grad():
            output = evaluate(line_to_tensor(n_letters, input_line), rnn)

            # Get top N categories
            topv, topi = output.topk(n_predictions, 1, True)
            predictions = []

            for i in range(n_predictions):
                value = topv[0][i].item()
                category_index = topi[0][i].item()
                print('(%.2f) %s' % (value, all_categories[category_index]))
                predictions.append([value, all_categories[category_index]])
        res[input_line] = predictions

    res_df = pd.DataFrame(res)
    res_df.to_csv(args.output_path)
    return res_df


def evaluate(line_tensor, rnn):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output


def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for character-rnn")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--train-file-path", type=str, default="data/names/*.txt",
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training names")
    train_arg_parser.add_argument("--learning-rate", type=float, default=0.005,
                                  help="learning rate for training, default is 0.005")
    train_arg_parser.add_argument("--n-hidden", type=int, default=128,
                                  help="size of training hidden state, default is 128")
    train_arg_parser.add_argument("--n-iters", type=int, default=5000,
                                  help="size of training iterations, default is 5000")
    train_arg_parser.add_argument("--save-model-path", type=str, required=True,
                                  help="path to folder where trained model will be saved.")

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/country name prediction arguments")
    eval_arg_parser.add_argument("--n-predictions", type=int, default=3,
                                 help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for predicting the names. If file ends in .pth - PyTorch path is used")
    eval_arg_parser.add_argument("--input-lines", type=str, required=True,
                                 help="input names for prediction")
    eval_arg_parser.add_argument("--output-path", type=str, required=True,
                                 help="output predictions for namecountries")

    args = main_arg_parser.parse_args()

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

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)

    if args.subcommand == "train":
        all_losses = train_batch(args)[0]
        return all_losses

    else:
        predict_country_name(args, all_categories, n_letters)


if __name__ == "__main__":
    # all_losses = main()
    # print(all_losses[:5])
    # plt.plot(all_losses)
    # plt.xlabel("iteration")
    # plt.ylabel("losses")
    # plt.title('training losses for basic RNN')
    # plt.savefig("data/plots/loss_rnn.png")
    main()