import datetime
import os

import numpy as np
import pywt
import wfdb
import tensorflow as tf
from tensorflow import keras
from utils_fourLayerCNN import loadData, plot_history_tf, plot_heat_map

#from ECG_read_weiyuqing_without_wfdb import ECGDATAPATH
#PATH of this test
Project_PATH = "../Number-Of-CNN-Layers/Four/"
#PICTUREPATH= Project_PATH + "picture/"
log_dir = Project_PATH + "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_Path_one = Project_PATH  + "model/ecg_model_four_layer.h5"
#model_Path_one = Project_PATH + "ecg_model_one_layer.h5"

RATIO = 0.3
RANDOM_SEED = 42
BATCH_SIZE = 128
NUM_EPOCHS = 10


def CNN_model_level_one():
    leavlOneModel = tf.keras.models.Sequential([


        #take test for two CNN layers model

        tf.keras.layers.InputLayer(input_shape=(300,)),
        tf.keras.layers.Reshape(target_shape=(300, 1)),

        tf.keras.layers.Conv1D(filters=15, kernel_size=3, strides=2, padding='same', activation='relu'),
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same'),
        tf.keras.layers.Conv1D(filters=15, kernel_size=3, strides=2, padding='same', activation='relu'),
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same'),
        tf.keras.layers.Conv1D(filters=15, kernel_size=3, strides=2, padding='same', activation='relu'),
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same'),
        tf.keras.layers.Conv1D(filters=15, kernel_size=3, strides=2, padding='same', activation='relu'),
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same'),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(5, activation='softmax'),
        # tf.keras.layers.Dropout(rate=0.2),
        # tf.keras.layers.Dense(5, activation='softmax')
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
