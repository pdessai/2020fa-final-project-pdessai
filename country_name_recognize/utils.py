from io import open
import glob
import os
import matplotlib.pyplot as plt
import unicodedata
import string
import random

import torch
import torch.nn as nn


all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


def find_files(path):
    return glob.glob(path)


def unicode_2_Ascii(s):
    """
    # Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
    """
    all_letters = string.ascii_letters + " .,;'"
    n_letters = len(all_letters)
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def read_lines(filename):
    """
    # Read a file and split into lines
    """
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicode_2_Ascii(line) for line in lines]


def letter_to_index(all_letters, letter):
    """
    # Find letter index from all_letters, e.g. "a" = 0
    """
    return all_letters.find(letter)


def letter_to_tensor(all_letters, n_letters, letter):
    """
    # Just for demonstration, turn a letter into a <1 x n_letters> Tensor
    """
    tensor = torch.zeros(1, n_letters)
    tensor[0][letter_to_index(all_letters,letter)] = 1
    return tensor


def line_to_tensor(n_letters, line):
    """
    # Turn a line into a <line_length x 1 x n_letters>,
    # or an array of one-hot letter vectors
    """
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(all_letters,letter)] = 1
    return tensor


def category_from_output(all_categories, output):
    """
    convert the output tensor to country names
    """
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def sample(l):
    return l[random.randint(0, len(l) - 1)]


def sample_trainning(n_letters, all_categories, category_lines):
    """
    generate random training data
    """
    category = sample(all_categories)
    line = sample(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(n_letters, line)
    return category, line, category_tensor, line_tensor


def train(rnn, criterion, learning_rate, category_tensor, line_tensor, model_type):
    if model_type == 'lstm':
        hx, cx = rnn.initHidden()

        rnn.zero_grad()

        for i in range(line_tensor.size()[0]):
            output, hx, cx = rnn(line_tensor[i], hx, cx)

    else:
        hidden = rnn.initHidden()

        rnn.zero_grad()

        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()


class RNN(nn.Module):
    """
    define a basic RNN network
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


class RNN_multi_1ayer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN_multi_1ayer, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)

        self.i2o = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o_2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output_2 = self.i2o_2(output)
        output_2 = self.softmax(output_2)
        return output_2, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = 1
        self.rnn = nn.LSTMCell(input_size=input_size,
                               hidden_size=hidden_size)

        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hx, cx):
        # hx, cx = self.initHidden()
        hx, cx = self.rnn(input, (hx, cx))
        # out = out.contiguous().view(-1, self.hidden_size)
        output = self.fc(hx)
        pred = self.softmax(output)
        return pred, hx, cx

    def initHidden(self):
        hx = torch.zeros(1, self.hidden_size)
        cx = torch.zeros(1, self.hidden_size)
        return hx, cx