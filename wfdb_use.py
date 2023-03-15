import os

import numpy as np
import wfdb
import matplotlib.pyplot

PATH = './mit-bih-arrhythmia-database-1.0.0/'


def read_ecg_data(data):
    # get the data from the ecg data file and set begin and end points
    record = wfdb.rdrecord(PATH + data, sampfrom=0, sampto=1500)
    # return type record
    print(type(record))
    print(dir(record))
    # get the Lead signal （len、filename、number、signalname、Sample rate
    print(record.p_signal)
    print(np.shape(record.p_signal))

    print(record.sig_len)

    print(record.record_name)

    print(record.n_sig)

    print(record.sig_name)

    print(record.fs)

    annotations = wfdb.rdann(PATH+data,'atr')
    print(type(annotations))
    print(dir(annotations))

    #get R
    print(annotations.sample)
    #get label of Heart beat
    print(annotations.sample)
    #num
    print(annotations.ann_len)
    #show label of beat
    print(wfdb.show_ann_labels())

    matplotlib.pyplot.draw_ecg(record.p_signal)
    return record.p_signal

