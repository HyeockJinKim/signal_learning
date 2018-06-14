import os
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import wave
from scipy import signal
from numpy.lib import stride_tricks

folder_path = 'modulation'
data_set = {}

# Amplitude 이미지를 만들어 냄.
file_path = os.path.join(folder_path, '4FSK', '152chunk1.wav')
sr, data = wavfile.read(file_path)
print(data)
time = np.linspace(0, len(data)/sr, num=len(data))
plt.figure(1)
plt.title('Signal Wave...')
plt.plot(time, data)
plt.show()

spec = np.fft.fft(data)
freq = np.fft.fftfreq(data.size, 1/sr)
mask = freq > 0
plt.figure(1)
plt.title('FFT Wave...')
plt.plot(freq[mask], np.abs(spec[mask]))
plt.show()

# 로그 스펙트로그램을 계산하는 함수 ( 주파수 )
def log_specgram(audio, sample_rate, window_size=20, step_size=10,
                 eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    # 연속 푸리에 변환을 해 스펙트로그램을 계산하는 함수
    freqs, times, spec = signal.spectrogram(audio,
                                            fs=sample_rate,
                                            window='hann',
                                            # 여기 윈도우 매개변수에 들어간 부분이
                                            # window function들 중에 하나이다.
                                            # 시가눅에서 Window Function을 곱하는 것은, 주파수 축에서 Filtering이라고 한단다.
                                            # Window의 주파수 스펙트럼을, 원래 입력 신호의 주파수 스펙트럼과 convolution해서 조금 더 부드럽고
                                            # side lobe가 사라진 주파수 스펙트럼을 만들 수 있다. But 단점으로는 Main lobe가 두꺼워진다는 점.
                                            # Cosine, Raised Cosine, Hamming, Hanning, Blackman, Triangular, Gaussian 등이 있다.
                                            # nperseg=nperseg,
                                            #
                                            # noverlap=noverlap,
                                            # 각 윈도우들을 겹치게 할지를 결정하는 매개변수
                                            detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)


# 주파수, 시간, 스펙트로그램을 Sample 들과, Sample_Rate 에 대해 값들을 계산한다.
freq, times, spectrogram = log_specgram(data, sr)

plt.figure(1)
plt.title('Signal Spectrogram...')
plt.imshow(spectrogram.T, aspect='auto', origin='lower',
           extent=[times.min(), times.max(), freq.min(), freq.max()])
plt.show()

print(len(spectrogram))
print(len(data))


def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)


def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            newspec[:, i] = np.sum(spec[:,int(scale[i]):], axis=1)
        else:
            newspec[:, i] = np.sum(spec[:,int(scale[i]):int(scale[i+1])], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[int(scale[i]):])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i+1])])]

    return newspec, freqs


""" plot spectrogram"""
def plotstft(audiopath, binsize=2**10, plotpath=None, colormap="jet"):
    samplerate, samples = wavfile.read(audiopath)

    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)

    ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel

    timebins, freqbins = np.shape(ims)

    print("timebins: ", timebins)
    print("freqbins: ", freqbins)

    plt.figure(figsize=(15, 7.5))
    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    plt.colorbar()

    plt.xlabel("time (s)")
    plt.ylabel("frequency (hz)")
    plt.xlim([0, timebins-1])
    plt.ylim([0, freqbins])

    xlocs = np.float32(np.linspace(0, timebins-1, 5))
    plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate])
    ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
    plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])

    if plotpath:
        plt.savefig(plotpath, bbox_inches="tight")
    else:
        plt.show()

    plt.clf()

    return ims


ims = plotstft(file_path)

