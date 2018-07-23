import os
from os.path import isdir, join
# Math
import random

import numpy as np
from scipy import signal
from scipy.io import wavfile
from cnn import Spectrogram_CNN

# Visualization
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import seaborn as sn


class SpectrogramLearning:

    def __init__(self, path):
        self.path = os.path.join(path)
        self.dirs = [f for f in os.listdir(path) if isdir(join(path, f))]
        self.dirs.sort()
        self.acc_loss_list = []
        self.confusion_matrix = None
        self.max_sr = None

        self.target_dict = {}
        self.target_all = []
        self.target_all2 = []
        self.target_value = {}
        self.spec_all = None
        self.phase_all = []
        self.freq_size = None
        self.time_size = None
        self.train_target = None
        self.train_spec = None
        self.train_spec2 = None
        self.test_target = None
        self.test_spec = None
        self.test_spec2 = None
        self.train_target2 = None
        self.test_target2 = None
        # call the init variable function
        self.reset_confusion_matrix()
        self.find_max_sr()

    @staticmethod
    def log_spectrogram(audio, sample_rate, window_size=20, step_size=10,
                        eps=1e-10, is_phase=False):
        nperseg = int(round(window_size * sample_rate / 1e3))
        noverlap = int(round(step_size * sample_rate / 1e3))
        if is_phase:
            fft = np.fft.fft(audio)

            freqs, times, spec = signal.spectrogram(np.imag(fft),
                                                    fs=sample_rate,
                                                    window='hann',
                                                    nperseg=nperseg,
                                                    noverlap=noverlap,
                                                    detrend=False)
        else:
            freqs, times, spec = signal.spectrogram(audio,
                                                    fs=sample_rate,
                                                    window='hann',
                                                    nperseg=nperseg,
                                                    noverlap=noverlap,
                                                    detrend=False)
        return freqs, times, np.log(spec.T.astype(np.float32) + eps)

    # Confusion Matrix
    def reset_confusion_matrix(self):
        self.confusion_matrix = np.zeros(shape=(len(self.dirs), len(self.dirs)))

    def find_max_sr(self):
        max_sr = 0
        for direct in self.dirs:
            sr, data = wavfile.read(join(self.path, direct, os.listdir(join(self.path, direct))[0]))
            if max_sr < sr:
                max_sr = sr
        self.max_sr = max_sr

    def find_min_num(self):
        min_num = len(os.listdir(os.path.join(self.path, self.dirs[0])))
        for direct in self.dirs:
            waves = [f for f in os.listdir(join(self.path, direct)) if f.endswith('.wav')]
            num = 0
            for wav in waves:
                try:
                    sample_rate, samples = wavfile.read(os.path.join(self.path, direct, wav))
                    num += 1
                except:
                    pass
            if min_num > num:
                min_num = num
        return min_num

    def read_wav(self):
        al = []
        phase_al = []
        min_num = self.find_min_num()
        min_num = 30
        print(min_num)
        for i, direct in enumerate(self.dirs):
            waves = [f for f in os.listdir(join(self.path, direct)) if f.endswith('.wav')]
            random.shuffle(waves)
            self.target_value[direct] = i
            self.target_dict[i] = direct
            print(str(i) + ':' + str(direct) + ' ', end='')
            num = 0
            for wav in waves:
                try:  # 읽지 못하는 파일 건너뜀.
                    sample_rate, samples = wavfile.read(os.path.join(self.path, direct, wav))
                    resamples = signal.resample(samples, self.max_sr)

                    freqs, times, spec = self.log_spectrogram(resamples, self.max_sr)
                    freqs, times, phase_spec = self.log_spectrogram(resamples, self.max_sr, is_phase=True)

                    spec = (spec - spec.min()) / (spec.max() - spec.min())
                    phase_spec = (phase_spec - phase_spec.min()) / (phase_spec.max() - phase_spec.min())

                    al.append([np.reshape(spec, (len(freqs), len(times))), direct])
                    phase_al.append([np.reshape(phase_spec, (len(freqs), len(times))), direct])
                    num += 1
                    if num >= min_num:
                        break
                except:
                    pass
        self.freq_size = len(freqs)
        self.time_size = len(times)
        np.random.shuffle(al)
        np.random.shuffle(phase_al)
        # Split Data to Spectrogram and Target(Label)
        self.spec_all = np.reshape(np.delete(al, 1, 1), (len(al)))
        self.target_all = [i for i in np.delete(al, 0, 1).tolist()]
        self.phase_all = np.reshape(np.delete(phase_al, 1, 1), (len(al)))
        self.target_all2 = [i for i in np.delete(phase_al, 0, 1).tolist()]

    def get_accuracy(self, logits, targets):
        batch_predictions = np.argmax(logits, axis=1)
        self.make_confusion_matrix(batch_predictions, targets)
        num_correct = np.sum(np.equal(batch_predictions, targets))
        return (100. * num_correct) / batch_predictions.shape[0]

    def make_confusion_matrix(self, pred, target):
        for i in range(len(pred)):
            self.confusion_matrix[pred[i]][target[i]] += 1

    def learning_cnn(self, lr, generations, drop_out_rate, batch_size):
        x_input, y_target, eval_input, eval_target, z_input = self.define_value(batch_size)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            with tf.variable_scope('scope', reuse=tf.AUTO_REUSE) as scope:
                model_output = Spectrogram_CNN.cnn_model(x_input, batch_size,
                                                         drop_out_rate=drop_out_rate,
                                                         is_training=True, target_value=self.target_value)
                model_output2 = Spectrogram_CNN.cnn_model(z_input, batch_size,
                                                         drop_out_rate=drop_out_rate,
                                                         is_training=True, target_value=self.target_value)
                test_model_output = Spectrogram_CNN.cnn_model(eval_input, batch_size, target_value=self.target_value)

            targets = tf.squeeze(tf.cast(y_target, tf.int32))
            # Define loss function Reduce_mean
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model_output, labels=y_target))
            prediction = tf.nn.softmax(model_output)
            loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model_output2, labels=y_target))
            prediction2 = tf.nn.softmax(model_output2)
            test_prediction = tf.nn.softmax(test_model_output)
            # my_optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=10e-2)
            # my_optimizer2 = tf.train.AdamOptimizer(learning_rate=lr*0.1, epsilon=10e-2)
            # my_optimizer = tf.train.AdamOptimizer(learning_rate = lr)

            my_optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
            my_optimizer2 = tf.train.MomentumOptimizer(learning_rate=lr*0.1, momentum=0.9)
            train_step = my_optimizer.minimize(loss)
            train_step2 = my_optimizer2.minimize(loss2)
            # Initializing global_variables
            init = tf.global_variables_initializer()
            sess.run(init)
            train_loss = []
            train_acc = []
            test_acc = []
            try:
                for i in range(generations):
                    rand_index = np.random.choice(len(self.train_spec), size=batch_size)
                    rand_x = self.train_spec[rand_index]
                    rand_x = np.expand_dims(rand_x, -1)
                    rand_z = self.train_spec2[rand_index]
                    rand_z = np.expand_dims(rand_z, -1)
                    rand_y = self.train_target[rand_index]
                    rand_y2 = self.train_target2[rand_index]

                    sess.run(train_step, feed_dict={x_input: rand_x, y_target: rand_y})
                    temp_train_loss, temp_train_preds = sess.run([loss, prediction],
                                                                 feed_dict={x_input: rand_x, y_target: rand_y})
                    temp_train_acc = self.get_accuracy(temp_train_preds, rand_y)

                    sess.run(train_step2, feed_dict={z_input: rand_z, y_target: rand_y2})
                    temp_train_loss, temp_train_preds = sess.run([loss2, prediction2],
                                                                 feed_dict={z_input: rand_z, y_target: rand_y2})
                    temp_train_acc = self.get_accuracy(temp_train_preds, rand_y2)
                    # logging temp result
                    if (i + 1) % 50 == 0:
                        eval_index = np.random.choice(len(self.test_spec), size=batch_size)
                        eval_x = self.test_spec[eval_index]
                        eval_x = np.expand_dims(eval_x, -1)
                        eval_y = self.test_target[eval_index]
                        eval_z = self.test_spec2[eval_index]
                        eval_z = np.expand_dims(eval_z, -1)
                        eval_y2 = self.test_target2[eval_index]

                        test_preds = sess.run(test_prediction, feed_dict={eval_input: eval_x})
                        temp_test_acc = self.get_accuracy(test_preds, eval_y)

                        # Logging and Printing Results
                        train_loss.append(temp_train_loss)
                        train_acc.append(temp_train_acc)
                        test_acc.append(temp_test_acc)
                        acc_and_loss = [(i + 1), temp_train_loss, temp_train_acc, temp_test_acc]
                        acc_and_loss = [np.round(x, 10) for x in acc_and_loss]
                        self.acc_loss_list.append('Generation # {}. Train Loss: {:.10f}. '
                                                  'Train Acc (Test Acc): {:.2f} ({:.2f})'.format(*acc_and_loss))

                        print('Generation # {}. Train Loss: {:.10f}. '
                              'Train Acc (Test Acc): {:.2f} ({:.2f})'.format(*acc_and_loss))

                        test_preds = sess.run(test_prediction, feed_dict={eval_input: eval_z})
                        temp_test_acc = self.get_accuracy(test_preds, eval_y2)

                        # Logging and Printing Results
                        train_loss.append(temp_train_loss)
                        train_acc.append(temp_train_acc)
                        test_acc.append(temp_test_acc)
                        acc_and_loss = [(i + 1), temp_train_loss, temp_train_acc, temp_test_acc]
                        acc_and_loss = [np.round(x, 10) for x in acc_and_loss]
                        self.acc_loss_list.append('Generation # {}. Train Loss: {:.10f}. '
                                                  'Train Acc (Test Acc): {:.2f} ({:.2f})'.format(*acc_and_loss))

                        print('Generation # {}. Train Loss: {:.10f}. '
                              'Train Acc (Test Acc): {:.2f} ({:.2f})'.format(*acc_and_loss))
            except KeyboardInterrupt:
                print('Stop!!')
                pass
            self.reset_confusion_matrix()

            for loop in range(5):
                no_list = []
                num_list = []
                for loops in range(3):
                    target_list = [index for index in self.target_dict.keys()]
                    while target_list:
                        rand_index = int(random.randrange(1, len(self.train_target)))
                        for i, index in enumerate(self.train_target):
                            for p in range(rand_index):
                                continue
                            num = int(index)
                            if num in target_list and i not in no_list:
                                target_list.remove(num)
                                no_list.append(i)
                                num_list.append(i)

                num_list.append(int(random.randrange(1, len(self.train_target))))
                num_list.append(int(random.randrange(1, len(self.train_target))))
                rand_x = self.train_spec[np.array(num_list)]
                rand_x = np.expand_dims(rand_x, -1)
                rand_y = self.train_target[np.array(num_list)]
                rand_z = self.train_spec2[np.array(num_list)]
                rand_z = np.expand_dims(rand_z, -1)
                rand_y2 = self.train_target2[np.array(num_list)]

                sess.run(train_step, feed_dict={x_input: rand_x, y_target: rand_y})
                temp_train_loss, temp_train_preds = sess.run([loss, prediction],
                                                             feed_dict={x_input: rand_x, y_target: rand_y})
                temp_train_acc = self.get_accuracy(temp_train_preds, rand_y)
                sess.run(train_step2, feed_dict={z_input: rand_z, y_target: rand_y2})
                temp_train_loss, temp_train_preds = sess.run([loss2, prediction2],
                                                             feed_dict={z_input: rand_z, y_target: rand_y2})
                temp_train_acc = self.get_accuracy(temp_train_preds, rand_y2)
                no_list = []
                num_list = []
                for loop in range(3):
                    target_list = [index for index in self.target_dict.keys()]
                    while target_list:
                        rand_index = int(random.randrange(1, len(self.test_target)))
                        for i, index in enumerate(self.test_target):
                            for p in range(rand_index):
                                continue
                            num = int(index)
                            if num in target_list and i not in no_list:
                                target_list.remove(num)
                                no_list.append(i)
                                num_list.append(i)

                num_list.append(int(random.randrange(1, len(self.test_target))))
                num_list.append(int(random.randrange(1, len(self.test_target))))

                eval_x = self.test_spec[np.array(num_list)]
                eval_x = np.expand_dims(eval_x, -1)
                eval_y = self.test_target[np.array(num_list)]
                eval_z = self.test_spec2[np.array(num_list)]
                eval_z = np.expand_dims(eval_z, -1)
                eval_y2 = self.test_target2[np.array(num_list)]

                test_preds = sess.run(test_prediction, feed_dict={eval_input: eval_x})
                temp_test_acc = self.get_accuracy(test_preds, eval_y)

                test_preds = sess.run(test_prediction, feed_dict={eval_input: eval_z})
                temp_test_acc = self.get_accuracy(test_preds, eval_y2)

    def define_value(self, batch_size):
        train_indices = np.random.choice(len(self.target_all), round(len(self.target_all) * 0.7), replace=False)

        # Get indices for test 30%
        test_indices = np.array(list(set(range(len(self.target_all))) - set(train_indices)))

        spec_vals = np.array([x for x in self.spec_all])
        target_vals = np.array([x for x in self.target_all])
        phase_vals = np.array([x for x in self.phase_all])
        target_vals2 = np.array([x for x in self.target_all2])

        self.train_spec = spec_vals[train_indices][:]
        train_target = target_vals[train_indices][:]
        self.train_spec2 = phase_vals[train_indices][:]
        train_target2 = target_vals2[train_indices][:]

        self.test_spec = spec_vals[test_indices][:]
        self.test_spec2 = phase_vals[test_indices][:]
        test_target = target_vals[test_indices][:]
        test_target2 = target_vals2[test_indices][:]

        temp = []
        for v in train_target:
            temp.append(self.target_value[v[0]])
        self.train_target = np.array(temp)

        temp = []
        for v in test_target:
            temp.append(self.target_value[v[0]])
        self.test_target = np.array(temp)

        temp = []
        for v in train_target2:
            temp.append(self.target_value[v[0]])
        self.train_target2 = np.array(temp)

        temp = []
        for v in test_target2:
            temp.append(self.target_value[v[0]])
        self.test_target2 = np.array(temp)

        x_input_shape = (batch_size, self.freq_size, self.time_size, 1)
        y_input_shape = (batch_size,)
        x_input = tf.placeholder(tf.float32, shape=x_input_shape)
        z_input = tf.placeholder(tf.float32, shape=x_input_shape)

        y_target = tf.placeholder(tf.int32, shape=y_input_shape)
        eval_input = tf.placeholder(tf.float32, shape=x_input_shape)
        eval_target = tf.placeholder(tf.int32, shape=y_input_shape)

        return x_input, y_target, eval_input, eval_target, z_input

    def show_confusion_matrix(self):
        index_list = [index for index in self.target_dict.items()]
        df_cm = pd.DataFrame(self.confusion_matrix,
                             index=index_list,
                             columns=index_list)
        plt.figure()
        sn.heatmap(df_cm, annot=True)
        plt.show()

    def save_result(self):
        save_path = os.path.join('cnn', 'save')
        with open(os.path.join(save_path, 'save_'+str(len(os.listdir(save_path)))+'_filter_big_3l'), 'w') as f:
            for ac_los in self.acc_loss_list:
                f.write(ac_los+'\n')
            f.write('cnf_matrix\n')
            for line in self.confusion_matrix:
                for v in line:
                    f.write(str(v)+' ')
                f.write('\n')


if __name__ == '__main__':
    spec_learn = SpectrogramLearning('peak_modulation')
    spec_learn.read_wav()
    lr = 0.001
    generations = 20000
    # num_gens_to_wait = 250
    drop_out_rate = 0.05
    batch_size = 32
    spec_learn.learning_cnn(lr, generations, drop_out_rate, batch_size)
    spec_learn.show_confusion_matrix()
    spec_learn.save_result()
