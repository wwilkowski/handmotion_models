import json
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import tensorflowjs as tfjs

NO_OF_EPOCHS = 10
TEST_SIZE = 0.3

if __name__ == '__main__':
    with open("./data.json") as f:
        data = json.load(f)
        X = []
        Y = []
        for i in range(len(data)):
            X.append(data[i]["xs"])
            Y.append(data[i]["ys"])
    X = np.array(X)
    Y = np.array(Y)
    Y = to_categorical(Y, 6)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE)

    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=([5])))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(6, activation='softmax'))

    model.summary()
    adam = keras.optimizers.Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    history = model.fit(X_train, Y_train, epochs=NO_OF_EPOCHS, validation_data=(X_test, Y_test))

    loss_train = history.history['loss']
    epochs = range(1, NO_OF_EPOCHS+1)
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("charts/loss.png")

    original_labels = keras.backend.argmax(Y_test)
    score = keras.backend.argmax(model.predict(X_test))

    fig, ax = plt.subplots()
    plot_data = confusion_matrix(original_labels, score)
    ax.matshow(plot_data, vmax=len(X_test)/6)

    for (i, j), z in np.ndenumerate(plot_data):
        ax.text(j, i, '{:0.0f}'.format(z), ha='center', va='center')

    plt.savefig("charts/test_confusion_matrix.png")
    model.save("poses_model_{}-epochs".format(NO_OF_EPOCHS))
    tfjs.converters.save_keras_model(model, 'tfjs_model')
