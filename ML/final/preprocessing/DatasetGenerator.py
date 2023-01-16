import librosa
import os

import numpy as np

from settings import defaults
from sklearn.model_selection import train_test_split
import SpectrogramMaker as sm

class DatasetGenerator():

    def __init__(self):
        pass

    def load_samples(self, dataset_path=defaults.DATASET_PATH, labels=['0', '1']):
        '''
        Loads audio wav files from a given dataset path and prepares them to be used in
        form suitable for classical machine learning algorithms. Examples are loaded into
        `X` field which serves as a collection of training (and testing) examples while
        `y` field serves as a ground truth for the given examples.
        '''
        X = []
        y = []

        for label in labels:
            directory = dataset_path + '/' + label
            for filename in os.listdir(directory):
                f = os.path.join(directory, filename)
                x, _ = librosa.load(f, sr=None)
                arr = np.array(x) 
                X.append(arr)
                y.append(int(label))

        return X, y
        

    def split_into_train_and_test(self, X, y, test_size=0.33, val_size=0.5, generate_val=False):
        '''
        Splits given X examples and y labels into train and test datasets (and optionally validation 
        dataset if requested). Initially, samples from X dataset are split into train and test dataset
        where test dataset contains (default value) 33% of examples from X. After that, splitting is done
        on the test dataset to generate validation set with (default value) 50% of examples being contained
        in the validation dataset from the previously generated test dataset.
        '''
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        
        if generate_val:
            X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, val_size=0.5)
            y_train, y_val, y_test = np.array(y_train).reshape((-1,1)), np.array(y_val).reshape((-1,1)), np.array(y_test).reshape((-1,1))
            return X_train, y_train, X_test, y_test, X_val, y_val
        else:
            y_train, y_test = np.array(y_train).reshape((-1,1)), np.array(y_test).reshape((-1,1))
            return X_train, y_train, X_test, y_test


    def make_spectrogram_dataset(X_train, y_train, X_test, y_test, X_val=None, y_val=None):
        '''
        Generates spectrogram dataset from train, test and (optionally) validation datasets
        of a given waveform data. 
        '''
        spec_maker = sm()

        X_train_spec = list(map(spec_maker.generate_spectrogram, X_train))
        X_test_spec = list(map(spec_maker.generate_spectrogram, X_test))
        X_train = np.array(X_train_spec)
        X_test = np.array(X_test_spec)
        y_train = np.array(y_train).reshape((-1))
        y_test = np.array(y_test).reshape((-1))

        if X_val is not None:
            X_val_spec = list(map(spec_maker.generate_spectrogram, X_val))
            X_val = np.array(X_val_spec)
            y_val = np.array(y_val).reshape((-1))
            return X_train, y_train, X_test, y_test, X_val, y_val
        else:
            return X_train, y_train, X_test, y_test

