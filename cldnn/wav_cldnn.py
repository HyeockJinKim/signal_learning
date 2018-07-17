import os
from os.path import join, isdir
import tensorflow as tf
import numpy as np
from scipy import signal
from scipy.io import wavfile

from cldnn.keras_learning_cldnn import *
from read_wav_sequence import *


classes = []


def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy) + 1])
    yy1[np.arange(len(yy)), yy] = 1
    return yy1


# return train, test value
def pre_process_cldnn(path, is_fft=False):
    dirs = get_dirs(path)
    global classes
    classes = dirs
    if is_fft:
        signal_all, label_all = read_wav(path, dirs, is_fft)
    else:
        signal_all, label_all = read_wav(path, dirs)
    signal_stack = np.stack(signal_all)

    data_num = signal_stack.shape[0]
    train_num = data_num * 0.5

    np.random.seed(2018)
    train_idx = np.random.choice(range(0, data_num), size=int(train_num), replace=False)
    test_idx = list(set(range(0, data_num)) - set(train_idx))

    x_train = signal_stack[train_idx]
    x_test = signal_stack[test_idx]

    y_train = to_onehot(list(map(lambda x: label_all[x], train_idx)))
    y_test = to_onehot(list(map(lambda x: label_all[x], test_idx)))

    print('shape: ', x_train.shape[1:])
    return x_train, x_test, y_train, y_test


def learning_cldnn(x_train, x_test, y_train, y_test, dr):
    model, x_train, x_test = keras_learning(x_train, x_test, y_train, y_test, dr, classes)
    check_result(model, x_test, y_test, classes)


def learning_signal(path):
    x_train, x_test, y_train, y_test = pre_process_cldnn(path)
    learning_cldnn(x_train, x_test, y_train, y_test, dr=0.3)


def learning_signal_fft(path):
    x_train, x_test, y_train, y_test = pre_process_cldnn(path, is_fft=True)
    learning_cldnn(x_train, x_test, y_train, y_test, dr=0.3)


if __name__ == '__main__':
    data_path = os.path.join('peak_modulation')
    # learning_signal(data_path)
    learning_signal_fft(data_path)
