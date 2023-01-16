import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tf.keras.layers import Normalization, Input, Resizing, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tf.keras.models import Sequential

def define_model(X_train):
    input_shape = X_train[0].shape
    print('Input shape:', input_shape)
    num_labels = 2

    # Instantiate the `tf.keras.layers.Normalization` layer.
    norm_layer = Normalization()
    # Fit the state of the layer to the spectrograms
    # with `Normalization.adapt`.
    norm_layer.adapt(data=X_train)

    model = Sequential([
        Input(shape=input_shape),
        # Downsample the input.
        Resizing(32, 32),
        # Normalize.
        norm_layer,
        Conv2D(32, 3, activation='relu'),
        Conv2D(64, 3, activation='relu'),
        MaxPooling2D(),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.summary()
    return model

def compile(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()],
    )

def fit(model, X_train, y_train, X_val, y_val):
    EPOCHS = 10
    history = model.fit(
        x=X_train, y=y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=5),
    )

    return history

def plot_metrics(history):
    metrics = history.history
    plt.figure(figsize=(16,6))
    plt.subplot(1,2,1)
    plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.ylim([0, max(plt.ylim())])
    plt.xlabel('Epoch')
    plt.ylabel('Loss [CrossEntropy]')

    plt.subplot(1,2,2)
    plt.plot(history.epoch, 100*np.array(metrics['accuracy']), 100*np.array(metrics['val_accuracy']))
    plt.legend(['accuracy', 'val_accuracy'])
    plt.ylim([0, 100])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy [%]')
    plt.show()

def evaluate(model, X_test):
    model.evaluate(X_test, return_dict=True)

def predict(model, X_test):
    y_predicted = model.predict(X_test)
    plt.hist(y_predicted)
    plt.show()

    return y_predicted

def plot_confusion_matrix(y_test, y_pred):
    confusion_mtx = tf.math.confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx,
                xticklabels=['0', '1'],
                yticklabels=['0', '1'],
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()

def print_precision_recall(y_pred, y_test):
    TP = sum([1 if (a==1 and b==1) else 0 for a,b in zip(y_pred, y_test)])
    FP = sum([1 if (a==1 and b==0) else 0 for a,b in zip(y_pred, y_test)])
    FN = sum([1 if (a==0 and b==1) else 0 for a,b in zip(y_pred, y_test)])

    print(f'TP={TP}, FP={FP}, FN={FN}')

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    print(f'Precision: {(precision*100):.2f}%')
    print(f'Recall: {(recall*100):.2f}%')