from feature import log_melspectrogram
import librosa
from pylab import *
import os
import glob

K = 5
feature_height = 64

#去除中间帧附近帧中包含音符起始点的样本, 使用db，使用归一化处理, 去除高频特征
def data_iterator(n_mels, sample_rate, time_win_size):
    dir = '../data_test'
    ground_truth = '../ground_truth'
    fold_num = int(len([list for list in os.listdir(dir)]) / K)
    for i in range(K):
        train_data_file, test_data_file, train_labels_file, test_labels_file = [], [], [], []
        index = 0
        for list in os.listdir(dir):
            if index >= i * fold_num and index < (i + 1) * fold_num:
                test_data_file.append(os.path.join(dir, list))
                test_labels_file.append(os.path.join(ground_truth, list.split('.')[0] + '.GroundTruth.txt'))
            else:
                train_data_file.append(os.path.join(dir, list))
                train_labels_file.append(os.path.join(ground_truth, list.split('.')[0] + '.GroundTruth.txt'))
            index += 1
        train_data, train_labels = data_process(train_data_file, train_labels_file, n_mels, sample_rate, time_win_size)
        test_data, test_labels = data_process(test_data_file, test_labels_file, n_mels, sample_rate, time_win_size)
        yield train_data, train_labels, test_data, test_labels

def data_process(data_file, labels_file, n_mels, sample_rate, time_win_size):
    X, y = [], []
    for i in range(len(data_file)):
        feature = log_melspectrogram(data_file[i], sample_rate=sample_rate, n_mels=n_mels)[:feature_height]
        duration = librosa.get_duration(filename=data_file[i])
        time_len = duration / len(feature[0])
        with open(labels_file[i]) as fp:
            l = fp.read().split('\n')
            fp.close()
            onset_list = [float(l[i * 3]) for i in range(int(len(l) / 3))]
            index = 0
            for j in range(len(feature[0]) - time_win_size):
                data = feature[:, j: j + time_win_size]
                max_val = np.max(data)
                min_val = np.min(data)
                data = (data - min_val) / (max_val - min_val)
                X.append(data)
                while index < len(onset_list) and onset_list[index] < (j + int(time_win_size / 2)) * time_len:
                    index += 1
                if index >= len(onset_list):
                    y.append(0)
                    continue
                if onset_list[index] < (j + int(time_win_size / 2) + 1) * time_len:
                    y.append(1)
                    print('%f - %d' % (onset_list[index], 1 + j + int(time_win_size / 2)))
                else:
                    y.append(0)
            # sum += len(onset_list)
    invalid = [0 for i in range(len(X))]
    for i in range(2, len(X)):
        if y[i - 1] == 1 or y[i - 2] == 1:
            invalid[i] = 1
    for i in range(0, len(X) - 2):
        if y[i + 1] == 1 or y[i + 2] == 1:
            invalid[i] = 1
    data, labels = [], []
    for i in range(len(X)):
        if invalid[i] == 0:
            data.append(X[i])
            labels.append(y[i])
    return data, labels

def vertify_data_iterator(n_mels, sample_rate, time_win_size):
    dir = '../data_test'
    ground_truth = '../ground_truth'
    fold_num = int(len([list for list in os.listdir(dir)]) / K)
    for i in range(K):
        train_data_file, test_data_file, train_labels_file, test_labels_file = [], [], [], []
        index = 0
        for list in os.listdir(dir):
            if index >= i * fold_num and index < (i + 1) * fold_num:
                test_data_file.append(os.path.join(dir, list))
                test_labels_file.append(os.path.join(ground_truth, list.split('.')[0] + '.GroundTruth.txt'))
            else:
                train_data_file.append(os.path.join(dir, list))
                train_labels_file.append(os.path.join(ground_truth, list.split('.')[0] + '.GroundTruth.txt'))
            index += 1
        train_data, train_labels = vertify_data_process(train_data_file, train_labels_file, n_mels, sample_rate, time_win_size)
        test_data, test_labels = vertify_data_process(test_data_file, test_labels_file, n_mels, sample_rate, time_win_size)
        yield train_data, train_labels, test_data, test_labels

def vertify_data_process(data_file, labels_file, n_mels, sample_rate, time_win_size):
    X, y = [], []
    for i in range(len(data_file)):
        feature = log_melspectrogram(data_file[i], sample_rate=sample_rate, n_mels=n_mels)[:feature_height]
        duration = librosa.get_duration(filename=data_file[i])
        time_len = duration / len(feature[0])
        with open('./onset_predict/' + data_file[i].split('/')[2].split('.')[0] + '.txt') as fp:
            predict = fp.read().split('\n')[:-1]
            fp.close()
            predict = [float(predict[i]) for i in range(len(predict))]
            predict = [int(predict[i] / time_len) - int(time_win_size / 2) for i in range(len(predict))]
        with open(labels_file[i]) as fp:
            l = fp.read().split('\n')
            fp.close()
            onset_list = [float(l[i * 3]) for i in range(int(len(l) / 3))]
            index = 0
            p = 0
            for j in range(len(feature[0]) - time_win_size):
                if j < predict[p]:
                    continue
                data = feature[:, j: j + time_win_size]
                max_val = np.max(data)
                min_val = np.min(data)
                data = (data - min_val) / (max_val - min_val)
                X.append(data)
                while index < len(onset_list) and onset_list[index] < (j + int(time_win_size / 2)) * time_len:
                    index += 1
                if index >= len(onset_list):
                    y.append(0)
                    continue
                if onset_list[index] < (j + int(time_win_size / 2) + 1) * time_len:
                    y.append(1)
                    print('%f - %d' % (onset_list[index], 1 + j + int(time_win_size / 2)))
                else:
                    y.append(0)
                p += 1
                if p == len(predict):
                    break
    return X, y