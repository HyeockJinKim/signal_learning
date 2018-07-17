import os
import pickle

import keras
from keras import Input, Model
from keras.layers import ZeroPadding2D, Conv2D, Dropout, Reshape, Dense, Activation, LSTM
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

import pandas as pd
import seaborn as sn

save_path = ''


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(title)
    plt.show()


def keras_get_model(x_train, x_test, y_train, y_test, dr, classes):
    input_x = Input(shape=(1, 2, 128))

    input_x_padding = ZeroPadding2D((0, 3))(input_x)
    layer11 = Conv2D(50, 1, 6, activation="relu", name="conv11", init='glorot_uniform')(input_x_padding)
    layer11 = Dropout(dr)(layer11)

    layer11_padding = ZeroPadding2D((0, 4))(layer11)
    layer12 = Conv2D(50, 1, 6, activation="relu", name="conv12", init='glorot_uniform')(layer11_padding)
    layer12 = Dropout(dr)(layer12)

    layer12 = ZeroPadding2D((0, 3))(layer12)
    layer13 = Conv2D(50, 1, 10, activation="relu", name="conv13", init='glorot_uniform')(layer12)
    layer13 = Dropout(dr)(layer13)

    concat = keras.layers.concatenate([layer11, layer13])
    concat_size = list(np.shape(concat))
    input_dim = int(concat_size[-1] * concat_size[-2])
    timesteps = int(concat_size[-3])

    concat = Reshape((timesteps, input_dim))(concat)

    lstm_out = LSTM(50)(concat)

    layer_dense1 = Dense(256, activation='relu', init='he_normal', name="dense1")(lstm_out)
    layer_dropout = Dropout(dr)(layer_dense1)
    layer_dense2 = Dense(len(classes), init='he_normal', name="dense2")(layer_dropout)

    layer_softmax = Activation('softmax')(layer_dense2)
    output = Reshape([len(classes)])(layer_softmax)

    model = Model(inputs=input_x, outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary()
    return model


def keras_learning(x_train, x_test, y_train, y_test, dr, classes):
    model = keras_get_model(x_train, x_test, y_train, y_test, dr, classes)
    epochs = 10000
    global batch_size
    batch_size = 512
    x_train = np.reshape(x_train, (-1, 1, 2, 128))
    x_test = np.reshape(x_test, (-1, 1, 2, 128))
    global save_path
    save_path = os.path.join('cldnn', 'save', str(dr))
    x_train = x_train[0:323]
    x_test = x_test[0:324]
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    history = model.fit(x_train,
                        y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=2,
                        validation_data=(x_test, y_test),
                        callbacks=[
                            keras.callbacks.ModelCheckpoint(save_path, monitor='val_loss', verbose=0,
                                                            save_best_only=True,
                                                            mode='auto'),
                            # keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
                        ])
    plt.figure()
    plt.title('Training performance')
    plt.plot(history.epoch, history.history['loss'], label='train loss+error')
    plt.plot(history.epoch, history.history['val_loss'], label='val_error')
    plt.legend()
    plt.savefig('cldnn Training performance')
    plt.show()
    return model, x_train, x_test


def show_confusion_matrix(cnf_matrix, classes):
    df_cm = pd.DataFrame(cnf_matrix,
                         index=classes,
                         columns=classes)
    plt.figure()
    sn.heatmap(df_cm, annot=True)
    plt.show()


def check_result(model, x_test, y_test, classes):
    model.load_weights(save_path)  # LOAD WEIGHT
    score = model.evaluate(x_test, y_test, verbose=0, batch_size=batch_size)
    print('evaluate_score:', score)
    test_Y_hat = model.predict(x_test, batch_size=batch_size)

    pre_labels = []
    for x in test_Y_hat:
        tmp = np.argmax(x, 0)
        pre_labels.append(tmp)
    true_labels = []
    for x in y_test:
        tmp = np.argmax(x, 0)
        true_labels.append(tmp)

    kappa = cohen_kappa_score(pre_labels, true_labels)
    oa = accuracy_score(true_labels, pre_labels)
    kappa_oa = {}
    print('oa_all:', oa)
    print('kappa_all:', kappa)
    kappa_oa['oa_all'] = oa
    kappa_oa['kappa_all'] = kappa
    with open('results_all_lstm_d%s.dat' % str(0.3), 'wb') as fd:
        pickle.dump(('lstm', 0.3, kappa_oa), fd)
    cnf_matrix = confusion_matrix(true_labels, pre_labels)
    print(cnf_matrix)
    show_confusion_matrix(cnf_matrix, classes)

