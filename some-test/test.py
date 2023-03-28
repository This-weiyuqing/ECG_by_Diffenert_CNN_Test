import datetime
import os
import numpy as np
import tensorflow as tf
from utils_seventyOne_of_Filters import loadData, plot_history_tf, plot_heat_mapProject_PATH = "../some-test\def main():
    X_train, X_test, y_train, y_test = loadData(RATIO, RANDOM_SEED)
    if os.path.exists(model_Path_one):
        print(' get model in h5 file')
        model = tf.keras.models.load_model(filepath=model_Path_one)
    else:
        # create new model(if model unexists)
        model = CNN_model_level_one()
        model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
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