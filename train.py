import torch
from model import convnet1, lstm
from feature import log_melspectrogram
import os
import librosa
import numpy as np
import torch.optim as optim
import torch.nn as nn
import random
import matplotlib.pyplot as plt
from data_processor import data_iterator, vertify_data_iterator

K = 5

n_mels = 128
sample_rate = 44100
time_win_size = 9
feature_height = 64

def train_conv(batch_size, epochs):
    fold = 0
    for train_X, train_y, test_X, test_y in data_iterator(n_mels, sample_rate, time_win_size):
        if fold != 0:
            break

        num_batchs = int(len(train_X) / batch_size)

        net = convnet1()
        optimizer = optim.Adam(net.parameters(), lr=0.003, weight_decay=0.0001)
        criterion = nn.BCELoss()
        train_loss, test_loss = [], []
        for epoch in range(epochs):
            index = np.array([i for i in range(len(train_X))])
            np.random.shuffle(index)
            train_data = torch.tensor([train_X[i] for i in index], dtype=torch.float)
            train_labels = torch.tensor([train_y[i] for i in index], dtype=torch.float)
            for i in range(num_batchs):
                train_in = train_data[i * batch_size: (i + 1) * batch_size].view(batch_size, 1, feature_height, 9)
                train_out = net(train_in, istraining=True, minibatch=batch_size).squeeze()
                loss = criterion(train_out, train_labels[i * batch_size: (i + 1) * batch_size])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print("epoch [%d / %d], batch [%d / %d], loss: %f" % (epoch, epochs, i, num_batchs, loss.data))
                train_loss.append(loss.data)

                index = np.array([j for j in range(len(test_X))])
                np.random.shuffle(index)
                test_data = torch.tensor([test_X[j] for j in index], dtype=torch.float)
                test_labels = torch.tensor([test_y[j] for j in index], dtype=torch.float)
                test_out = net(test_data[: batch_size].view(batch_size, 1, feature_height, 9), istraining=False, minibatch=batch_size).squeeze()
                test_loss.append(criterion(test_out, test_labels[: batch_size]).data)

        torch.save(net.state_dict(), 'model' + str(fold) + '.pt')
        fold += 1
        plot_loss(train_loss, test_loss)

def train_lstm(batch_size, epochs):
    fold = 0
    for train_data, train_labels, test_X, test_y in vertify_data_iterator(n_mels, sample_rate, time_win_size):
        if fold != 0:
            break
        num_batchs = int(len(train_data) / batch_size)
        train_data = torch.tensor(train_data, dtype=torch.float)
        train_labels = torch.tensor(train_labels, dtype=torch.float)

        net = lstm()
        optimizer = optim.Adam(net.parameters(), lr=0.003, weight_decay=0.0001)
        criterion = nn.BCELoss()
        train_loss, test_loss = [], []
        for epoch in range(epochs):

            for i in range(num_batchs):
                train_in = train_data[i * batch_size: (i + 1) * batch_size].permute(0, 2, 1)
                train_out = net(train_in).squeeze()
                loss = criterion(train_out, train_labels[i * batch_size: (i + 1) * batch_size])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print("epoch [%d / %d], batch [%d / %d], loss: %f" % (epoch, epochs, i, num_batchs, loss.data))
            train_loss.append(loss.data)

            index = np.array([j for j in range(len(test_X))])
            np.random.shuffle(index)
            test_data = torch.tensor([test_X[j] for j in index], dtype=torch.float).permute(0, 2, 1)
            test_labels = torch.tensor([test_y[j] for j in index], dtype=torch.float)
            test_out = net(test_data[: batch_size]).squeeze()
            test_loss.append(criterion(test_out, test_labels[: batch_size]).data)
        torch.save(net.state_dict(), 'lstm_model' + str(fold) + '.pt')
        fold += 1
        plot_loss(train_loss, test_loss)

def plot_loss(train_loss, test_loss):
    plt.plot(np.array(train_loss), color='r')
    plt.plot(np.array(test_loss), color='b')
    plt.show()

def plot_onset(model, fold, threshold=0.004, path='../data_test/child1.wav'):
    feature = log_melspectrogram(path, sample_rate=sample_rate, n_mels=n_mels)[:feature_height]
    output = []
    for i in range(int(time_win_size / 2)):
        output.append(0)
    for i in range(len(feature[0]) - time_win_size):
        X = torch.tensor(feature[:, i: i + time_win_size], dtype=torch.float).view(1, 1, feature_height, 9)
        y = model(X, istraining=False).squeeze()
        output.append(y.item())

    for i in range(int(time_win_size / 2)):
        output.append(0)
    duration = librosa.get_duration(filename=path)
    time_unit = duration / len(output)
    x = [i * time_unit for i in range(len(output))]
    output = np.array(output)
    x = np.array(x)
    plt.plot(x, output)
    plt.show()

    with open('result' + str(fold) + '.txt', 'w') as f:
        for i in range(1, len(output) - 1):
            if(output[i] > threshold) and output[i] >= output[i - 1] and output[i] >= output[i + 1]:
                f.write(str((i + 0.5) * time_unit) + '\n')

        f.close()

def onset_p(model_name, fold):

    model = convnet1()
    model.load_state_dict(torch.load(model_name))
    plot_onset(model, fold)

# def train_rnn(batch_size, epochs):
