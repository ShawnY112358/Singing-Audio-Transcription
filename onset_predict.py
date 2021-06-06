import numpy as np
from feature import log_melspectrogram
import os
import torch
from model import convnet1
import librosa
import sys

MAX_NOTE_NUM=sys.maxsize

sample_rate = 44100
n_mels = 128
time_win_size = 9

peak_threshold = 0.008
shortest_note = 5   #最短音符帧长

def onset_predict(model_name, threshold=peak_threshold):

    model = convnet1()
    model.load_state_dict(torch.load(model_name))
    dir = './data_test'
    predict_dir = './onset_predict'
    ground_truth = '../ground_truth'
    if not os.path.exists(predict_dir):
        os.mkdir(predict_dir)

    for list in os.listdir(dir):
        path = os.path.join(dir, list)
        feature = log_melspectrogram(path, sample_rate=sample_rate, n_mels=n_mels)[:64]
        duration = librosa.get_duration(filename=path)
        time_unit = duration / len(feature[0])
        output = []
        for i in range(int(time_win_size / 2)):
            output.append(0)
        for i in range(len(feature[0]) - time_win_size):
            X = torch.tensor(feature[:, i: i + time_win_size], dtype=torch.float).view(1, 1, 64, 9)
            y = model(X, istraining=False).squeeze()
            output.append(y.item())
        valid_index = peak_pick(output, threshold)
        predict = [(valid_index[i] + 0.5) * time_unit for i in range(len(valid_index))]
        with open(os.path.join(predict_dir, list.split('.')[0] + '.txt'), 'w') as f:
            for p in predict:
                f.write(str(p) + '\n')
            f.close()

def peak_pick(output, threshold=peak_threshold):
    valid_peak = []
    for i in range(1, len(output) - 1):
        if(output[i] >= output[i - 1] and output[i] >= output[i + 1]) and output[i] > threshold:
            valid_peak.append(i)
    constant = []
    onset = []
    valid_peak.append(MAX_NOTE_NUM)
    for i in range(len(valid_peak) - 1):
        if valid_peak[i + 1] - valid_peak[i] < shortest_note:
            constant.append(valid_peak[i])
        else:
            if constant != []:
                constant.append(valid_peak[i])
                onset.append(constant[int(len(constant) / 2)])
                constant = []
            else:
                onset.append(valid_peak[i])

    return onset


