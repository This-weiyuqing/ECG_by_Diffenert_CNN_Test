def get_MAX():
    MAX = 0
    for i in range(0, len(X_train)):
        if MAX < len(X_train[i]):
            MAX = len(X_train[i])
    for i in range(0, len(X_test)):
        if MAX < len(X_test[i]):
            MAX = len(X_test[i])
    return MAX
def made_CNN():
    # 1D CNN neural network
    model = Sequential()
    model.add(Conv1D(32, 5, activation='relu', input_shape=(MAX, 1)))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(256, 5, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(512, 5, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.summary()
    return model