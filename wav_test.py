import os

from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt


folder_path = 'peak_modulation'
dirs = os.listdir(folder_path)

for dir in dirs:

    for f in os.listdir(os.path.join(folder_path, dir)):
        sr, data = wavfile.read(os.path.join(folder_path, dir, f))
        print(data)
        time = np.linspace(0, len(data) / sr, num=len(data))
        plt.figure(1)
        plt.title('Signal Wave...')
        plt.plot(time, data)
        plt.show()

