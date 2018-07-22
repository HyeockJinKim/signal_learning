
import matplotlib.pyplot as plt
import matplotlib.spines
from scipy.interpolate import spline, interp1d
import numpy as np
# 'Generation # 50. Train Loss: 2.3025851250. Train Acc (Test Acc): 12.50 (21.88)'


def check_result(path):
    gen = []
    loss = []
    acc = []
    test = []
    with open(path, 'r') as f:
        line = f.readline()
        while line is not None:
            try:
                for x in range(2):
                    line = line.split()
                    g = int(line[2].split('.')[0])
                    l = float(line[5].split('.')[0]+line[5].split('.')[1])
                    a = float(line[10])
                    t = float(line[11].split('(')[1].split(')')[0])
                    line = f.readline()
                gen.append(g)
                loss.append(l)
                acc.append(a)
                test.append(t)
            except:
                break
    return np.array(gen), np.array(loss), np.array(acc), np.array(test)


def make_smooth_plot(x, y):
    plt.figure(1)

    # x_new = np.linspace(np.min(x), np.max(x), 500)
    # f = interp1d(x, y, kind='quadratic')
    # y_smooth = f(x_new)
    # plt.plot(x_new, y_smooth)
    # plt.scatter(x, y)


if __name__ == '__main__':
    gen, loss, acc, test = check_result('save/save_36_dif_lr')
    make_smooth_plot(gen, acc)


