import os
from os.path import join, isdir

from scipy import signal
from scipy.io import wavfile
import numpy as np


def get_dirs(path):
    dirs = [f for f in os.listdir(path) if isdir(join(path, f))]
    dirs.sort()
    return dirs


def get_min_shape(dirs):
    min_shape = 48000
    print(dirs)
    for direct in os.listdir(dirs):
        try:
            sr, data = wavfile.read(join(dirs, direct, os.listdir(join(dirs, direct))[0]))
            if min_shape > np.shape(data)[0]:
                min_shape = np.shape(data)[0]
        except:
            print('No Data')
    return min_shape


def mod_shape(shape):
    while shape % 256 != 0:
        shape -= 1
    return shape


def read_wav(path, dirs, is_fft=False):
    min_shape = get_min_shape(path)
    min_shape = mod_shape(min_shape)
    signal_all = []
    label_all = []
    for i, direct in enumerate(dirs):
        waves = [f for f in os.listdir(join(path, direct)) if f.endswith('.wav')]
        print(str(i) + ':' + str(direct) + ' ', end='')
        for wav in waves:
            sample_rate, samples = wavfile.read(os.path.join(path, direct, wav))
            try:
                samples = samples[0:min_shape, 0]  # for OFDM 2 channels
            except:
                samples = samples[0:min_shape, ]
            if is_fft:
                signal_fft = np.fft.fft(samples)
                real = np.real(signal_fft)
                imagine = np.imag(signal_fft)
                signal_all.append(np.array((real, imagine)))
            else:
                signal_all.append(np.array(samples))
            label_all.append(i)
    print()
    return signal_all, label_all
