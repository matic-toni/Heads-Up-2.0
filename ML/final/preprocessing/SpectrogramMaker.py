import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

class SpectrogramMaker():

    def __init__(self):
        pass

    def plot_waveform(self, X, y, offset=0):
        '''
        Plots nine waveforms from the given dataset X and graphically renders them in
        3 by 3 grid with each of them being displayed alongside its ground truth value.
        '''
        rows = 3
        cols = 3
        n = rows * cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, 9))

        for i in range(n):
            if i>=n:
                break
            r = i // cols
            c = i % cols
            ax = axes[r][c]
            ax.plot(X[i+offset])
            ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
            label = y[i+offset]
            ax.set_title(str(label))
            ax.set_ylim([-1.1,1.1])

            plt.show()

    def get_spectrogram(self, waveform):
        '''
        Converts the waveform given as the function argument to a spectrogram
        via a STFT. 
        '''
        # Convert the waveform to a spectrogram via a STFT.
        spectrogram = tf.signal.stft(
            waveform, frame_length=255, frame_step=128)
        # Obtain the magnitude of the STFT.
        spectrogram = tf.abs(spectrogram)
        # Add a `channels` dimension, so that the spectrogram can be used
        # as image-like input data with convolution layers (which expect
        # shape (`batch_size`, `height`, `width`, `channels`).
        spectrogram = spectrogram[..., tf.newaxis]
        return spectrogram

    def plot_spectrogram(spectrogram, ax):
        '''
        Plots spectrogram alongside its waveform graphical visualization displayed
        in two rows for each of the components.
        '''
        if len(spectrogram.shape) > 2:
            assert len(spectrogram.shape) == 3
            spectrogram = np.squeeze(spectrogram, axis=-1)
        # Convert the frequencies to log scale and transpose, so that the time is
        # represented on the x-axis (columns).
        # Add an epsilon to avoid taking a log of zero.
        log_spec = np.log(spectrogram.T + np.finfo(float).eps)
        height = log_spec.shape[0]
        width = log_spec.shape[1]
        X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
        Y = range(height)
        ax.pcolormesh(X, Y, log_spec)