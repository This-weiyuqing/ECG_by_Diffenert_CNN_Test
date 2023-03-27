# this file give
import os

import wfdb
import pywt
import seaborn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# from _mian_tensorflow import Project_PATH
Project_PATH = '../Number_Of_Filters/Three/'

PICTUREPATH = Project_PATH + 'picture/'

# from ECG_read_weiyuqing_without_wfdb import ECGDATAPATH
MITBITECGDATAPATH = '../MIT-BIT/'


# wavelet denoise preprocess using mallat algorithm

def denoise(date):
    # Wavelet db5 level9
    coeffs = pywt.wavedec(data=date, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    # noice del
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    # take cD 1,2 turn 0 to denoise
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)
    # get signal that before wavelet
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata


# get ECG data,date type and denoise
# this function without return
# use append take return data to pass data
def getDataSet(number, X_data, Y_data):
    ecgClassSet = ['N', 'A', 'V', 'L', 'R']
    print('get' + number + ' ECG data')
    record = wfdb.rdrecord(MITBITECGDATAPATH + number, channel_names=['MLII'])
    data = record.p_signal.flatten()
    # use def denoise
    rdata = denoise(date=data)

    # get type of ECG data
    annotation = wfdb.rdann(MITBITECGDATAPATH + number, 'atr')
    Rlocation = annotation.sample  # annotation.sample:get R
    Rclass = annotation.symbol

    # delete stable date near R
    start = 10
    end = 5
    i = start
    j = len(annotation.symbol) - end

    # get NAVLR type data
    while i < j:
        try:
            lable = ecgClassSet.index(Rclass[i])
            x_train = rdata[Rlocation[i] - 99:Rlocation[i] + 201]
            X_data.append(x_train)
            Y_data.append(lable)
            i += 1
        except ValueError:
            i += 1
    return


# RATIO=The ratio of the training set to the test set，usually is 0.3
def loadData(ratio, random_seed):
    numberSet = ['100', '101', '103', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115',
                 '116', '117', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '208',
                 '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230',
                 '231', '232', '233', '234']
    dataSet = []
    labelSet = []
    for n in numberSet:
        getDataSet(n, dataSet, labelSet)

    # dataSet = np.array(dataSet).reshape(-1, 300)
    # labelSet = np.array(labelSet).reshape(-1, 1)

    # reshape the data and split the dataset
    dataSet = np.array(dataSet).reshape(-1, 300)
    lableSet = np.array(labelSet).reshape(-1)
    X_train, X_test, y_train, y_test = train_test_split(dataSet, lableSet, test_size=ratio, random_state=random_seed)
    # print(X_train, X_test, y_train, y_test)
    return X_train, X_test, y_train, y_test

    # train_ds = np.hstack((dataSet, labelSet))
    # np.random.shuffle(train_ds)
    #
    # X = train_ds[:, :300].reshape(-1, 300, 1)
    # Y = train_ds[:, 300]
    # # cut test and train data from ECG data
    # shuffle_index = np.random.permutation(len(X))
    # test_length = int(RATIO * len(shuffle_index))
    # test_index = shuffle_index[:test_length]
    # train_index = shuffle_index[test_index:]
    # X_test, Y_test = X[test_index:], Y[test_index]
    # X_train, Y_train = X[train_index], Y[train_index]
    # return X_train, Y_test, X_test, Y_train


# confusion matrix
def plot_heat_map(y_test, y_pred):
    con_mat = confusion_matrix(y_test, y_pred)
    # normalize
    # con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]
    # con_mat_norm = np.around(con_mat_norm, decimals=2)
    # os.remove(PICTUREPATH+'confusion_matrix.png')
    # plot
    plt.figure(figsize=(8, 8))
    seaborn.heatmap(con_mat, annot=True, fmt='.20g', cmap='Blues')
    plt.ylim(0, 5)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig(PICTUREPATH + 'confusion_matrix.png')
    plt.show()


def plot_history_tf(history):
    # os.remove(PICTUREPATH+'accuracy.png')
    # os.remove(PICTUREPATH+'loss.png')

    plt.figure(figsize=(8, 8))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(PICTUREPATH + 'accuracy.png')
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(PICTUREPATH + 'loss.png')
    plt.show()


def plot_history_torch(history):
    plt.figure(figsize=(8, 8))
    plt.plot(history['train_acc'])
    plt.plot(history['test_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(PICTUREPATH + 'accuracy.png')
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.plot(history['train_loss'])
    plt.plot(history['test_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(PICTUREPATH + 'loss.png')
    plt.show()
