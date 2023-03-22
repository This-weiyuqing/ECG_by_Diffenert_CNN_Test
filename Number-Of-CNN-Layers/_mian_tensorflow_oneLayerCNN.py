import datetime
import os

import numpy as np
import pywt
import wfdb
import tensorflow as tf
from tensorflow import keras
from utils_oneLayerCNN import loadData, plot_history_tf, plot_heat_map

#from ECG_read_weiyuqing_without_wfdb import ECGDATAPATH
#PATH of this test
Project_PATH = "../Number-Of-CNN-Layers/One/"
#PICTUREPATH= Project_PATH + "picture/"
log_dir = Project_PATH + "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_Path_one = Project_PATH  + "model/ecg_model_one_layer.h5"
#model_Path_one = Project_PATH + "ecg_model_one_layer.h5"

RATIO = 0.3
RANDOM_SEED = 42
BATCH_SIZE = 128
NUM_EPOCHS = 30


def CNN_model_level_one():
    leavlOneModel = tf.keras.models.Sequential([

        # #four CNN layers test
        # tf.keras.layers.InputLayer(input_shape=(300,)),
        # # reshape the tensor with shape (batch_size, 300) to (batch_size, 300, 1)
        # tf.keras.layers.Reshape(target_shape=(300, 1)),
        # # the first convolution layer, 4 21x1 convolution kernels, output shape (batch_size, 300, 4)
        # tf.keras.layers.Conv1D(filters=4, kernel_size=21, strides=1, padding='SAME', activation='relu'),
        # # the first pooling layer, max pooling, pooling size=3 , stride=2, output shape (batch_size, 150, 4)
        # tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='SAME'),
        # # the second convolution layer, 16 23x1 convolution kernels, output shape (batch_size, 150, 16)
        # tf.keras.layers.Conv1D(filters=16, kernel_size=23, strides=1, padding='SAME', activation='relu'),
        # # the second pooling layer, max pooling, pooling size=3, stride=2, output shape (batch_size, 75, 16)
        # tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='SAME'),
        # # the third convolution layer, 32 25x1 convolution kernels, output shape (batch_size, 75, 32)
        # tf.keras.layers.Conv1D(filters=32, kernel_size=25, strides=1, padding='SAME', activation='relu'),
        # # the third pooling layer, average pooling, pooling size=3, stride=2, output shape (batch_size, 38, 32)
        # tf.keras.layers.AvgPool1D(pool_size=3, strides=2, padding='SAME'),
        # # the fourth convolution layer, 64 27x1 convolution kernels, output shape (batch_size, 38, 64)
        # tf.keras.layers.Conv1D(filters=64, kernel_size=27, strides=1, padding='SAME', activation='relu'),
        # # flatten layer, for the next fully connected layer, output shape (batch_size, 38*64)
        # tf.keras.layers.Flatten(),
        # # fully connected layer, 128 nodes, output shape (batch_size, 128)
        # tf.keras.layers.Dense(128, activation='relu'),
        # # Dropout layer, dropout rate = 0.2
        # tf.keras.layers.Dropout(rate=0.2),
        # # fully connected layer, 5 nodes (number of classes), output shape (batch_size, 5)
        # tf.keras.layers.Dense(5, activation='softmax')

        #take test for one CNN layers model

        tf.keras.layers.InputLayer(input_shape=(300, 1)),

        # ECG data cant add 0 in edge, will broken data ,take false answer. So we need padding same data in edge.

        # take four  conv excitation function used RElu
        tf.keras.layers.Conv1D(filters=1, kernel_size=3, strides=1, padding='same', activation='relu'),
        # take four  pool,take strids 2.
        tf.keras.layers.MaxPool1D(pool_size=1, strides=2, padding='same'),

        # tf.keras.layers.Conv1D(filters=4, kernel_size=21, strides=1, padding='same', activation='relu'),
        # tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same'),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        # add dropout  layer, dropout rate = 0.2
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    return leavlOneModel


def main():
    # get traxin test
    X_train, X_test, y_train, y_test = loadData(RATIO, RANDOM_SEED)

    # get model or create model
    if os.path.exists(model_Path_one):
        print(' get model in h5 file')
        model = tf.keras.models.load_model(filepath=model_Path_one)
    else:
        # create new model(if model unexists)
        model = CNN_model_level_one()
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.summary()
        # TB make
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                              histogram_freq=1,
                                                              )
        # Training and validation
        history = model.fit(X_train, y_train, epochs=NUM_EPOCHS,
                            batch_size=BATCH_SIZE,
                            validation_data=(X_test, y_test),
                            callbacks=[tensorboard_callback])
        # validation_split=RATIO,
        model.save(filepath=model_Path_one)
        plot_history_tf(history)

    y_pred = np.argmax(model.predict(X_test), axis=-1)
    plot_heat_map(y_test, y_pred)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    main()
