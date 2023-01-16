import librosa
import os
import random
import tensorflow as tf

def get_spectrogram(waveform):
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

class ExportModel(tf.Module):
  def __init__(self, model):
    self.model = model

    # Accept either a string-filename or a batch of waveforms.
    # You could add additional signatures for a single wave, or a ragged-batch. 
    # self.__call__.get_concrete_function(
    #     x=tf.TensorSpec(shape=(), dtype=tf.string))
    # self.__call__.get_concrete_function(
    #    x=tf.TensorSpec(shape=[None, 16000], dtype=tf.float32))

  @tf.function(input_signature=[tf.TensorSpec(shape=[7527,], dtype=tf.float32)])
  def encode(self, x):
    # If they pass a string, load the file and decode it. 
    if x.dtype == tf.string:
      x = tf.io.read_file(x)
      x, _ = tf.audio.decode_wav(x, desired_channels=1, desired_samples=7527,)
      x = tf.squeeze(x, axis=-1)
      x = x[tf.newaxis, :]

    x = x[tf.newaxis, :]
    # maybe convert x to [-1, 1]
    x = get_spectrogram(x)  
    result = self.model(x, training=False)	
    return {
      'result': result,     
    }

  '''
  @tf.function(input_signature=[tf.TensorSpec(shape=[1, 1], dtype=tf.float32)])
  def decode(self, x):
    class_ids = tf.argmax(x, axis=-1)
    class_names = tf.gather(label_names, class_ids)
    return {
        'predictions': x
    }
  '''



export = ExportModel(model)

f = os.path.join(directory, filename)
x, s = librosa.load(f)
x = tf.constant(x)
encoded = export.encode(x) 
print(encoded)
# decoded = export.decode(encoded['result'])
# print(decoded)

module_with_signature_path = os.path.join('.', 'module_with_signature')
tf.saved_model.save(
    export, 
    module_with_signature_path, 
    signatures={
        'encode': export.encode.get_concrete_function(),
        # 'decode': export.decode.get_concrete_function()
    })


converter = tf.lite.TFLiteConverter.from_saved_model(module_with_signature_path)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
]
tflite_model = converter.convert()

interpreter = tf.lite.Interpreter(model_content=tflite_model)
signatures = interpreter.get_signature_list()
print(signatures)

with open('headsup_model.tflite', 'wb') as f:
  f.write(tflite_model)


# Load the TFLite model in TFLite Interpreter
interpreter = tf.lite.Interpreter('headsup_model.tflite')
# There are 2 signatures defined in the model.
# If there are multiple signatures then we can pass the name.
encode = interpreter.get_signature_runner('encode')
# decode = interpreter.get_signature_runner('decode')

# my_signature is callable with input as arguments.
encoded = encode(x=x)
# 'output' is dictionary with all outputs from the inference.
# In this case we have single output 'result'.
print(encoded)
# decoded = decode(x=encoded['result'])
# print(decoded)


interpreter = tf.lite.Interpreter(model_content=tflite_model)

signatures = interpreter.get_signature_list()
print('Signature:', signatures)

encode = interpreter.get_signature_runner('encode')
# decode = interpreter.get_signature_runner('decode')

encoded = encode(x=x)
print(encoded)
# decoded = decode(x=encoded['result'])
# print(decoded)
# do ode je dobro