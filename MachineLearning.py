import csv
import tensorflow as tf
import numpy as np

print("TensorFlow version:", tf.__version__)

with open('kaggle/traffic.csv') as file:
  reader = csv.reader(file)
  file = list(reader)

mnist = file

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])