import json
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs
from keras.layers import Dense, LSTM
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

if __name__ == '__main__':
    X = []
    Y = []
    maxNumber = 0
    maxX = 0
    maxY = 0

    with open("./data.json") as f:
        data_json = json.load(f)

        for i in range(len(data_json)):
            data_json[i]["xs"] = np.array(data_json[i]["xs"])

        for i in range(len(data_json)):
            data = data_json[i]["xs"]

            l = len(data)
            if l > maxNumber:
                maxNumber = l

            maxX = 0
            maxY = 0
            maxVerticalDistance = 0
            X_data = []
            for j in range(len(data)):
                X_data.append([data[j][0], data[j][1], data[j][3]])
            X.append(np.array(X_data))
            Y.append(data_json[i]["ys"])

    X = np.array(X)
    Y = np.array(Y)
    Y = to_categorical(Y, 6)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    full_X = []
    full_Y = []
    for i in range(len(X)):
        for j in range(3, len(X[i])):
            full_X.append(X[i][:j])
            full_Y.append(Y[i])

    for i in range(len(full_X)):
        maxX = 0
        maxY = 0
        maxVerticalDistance = 0
        for j in range(len(full_X[i])):
            if maxX < full_X[i][j][0]:
                maxX = full_X[i][j][0]
            if maxY < full_X[i][j][1]:
                maxY = full_X[i][j][1]
            if maxVerticalDistance < full_X[i][j][2]:
                maxVerticalDistance = full_X[i][j][2]
        full_X[i] = full_X[i] / [maxX, maxY, maxVerticalDistance]

    full_X_train, full_X_test, full_Y_train, full_Y_test = train_test_split(full_X, full_Y, test_size=0.2)

    maxTestLength = 0
    for i in full_X_test:
        if len(i) > maxTestLength:
            maxTestLength = len(i)
    full_X_train = tf.ragged.constant(full_X_train)
    full_X_test = tf.ragged.constant(full_X_test)

    full_Y_train = np.array(full_Y_train)
    full_Y_test = np.array(full_Y_test)

    model = Sequential()
    model.add(LSTM(128, input_shape=(None, 3), return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(6, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    no_of_epochs = 50

    history = model.fit(full_X_train, full_Y_train, epochs=no_of_epochs, shuffle=False,
                        validation_data=(full_X_test, full_Y_test))

    loss_train = history.history['loss']
    epochs = range(1, no_of_epochs + 1)
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("charts/training_loss.png")

    original_labels = keras.backend.argmax(full_Y_test)
    score = keras.backend.argmax(model.predict(full_X_test))

    fig, ax = plt.subplots()
    plot_data = confusion_matrix(original_labels, score)
    ax.matshow(plot_data, cmap=plt.cm.Blues, vmax=len(X_test) / 6)

    for (i, j), z in np.ndenumerate(plot_data):
        ax.text(j, i, '{:0.0f}'.format(z), ha='center', va='center')

    plt.savefig("charts/test_results.png")

    model.save("gestures-keras-model-{}-epochs".format(no_of_epochs))
    tfjs.converters.save_keras_model(model, 'tfjsmodel')
