#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# code copied from TensorFlow tutorial

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
print("TensorFlow version:", tf.__version__)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


# credit goes to https://medium.com/@mgazar/lenet-5-in-9-lines-of-code-using-keras-ac99294c8086 for code ideas
# model = tf.keras.Sequential([
#     tf.keras.layers.InputLayer(shape=(28, 28, 1)),
#     tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same', activation='tanh'),
#     tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(14, 14)),
#     tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid', activation='tanh'),
#     tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(5, 5)),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(units=120, activation='tanh'),
#     tf.keras.layers.Dense(units=84, activation='tanh'),
#     tf.keras.layers.Dense(units=10, activation='softmax')
# ])

# god save this video
# https://www.youtube.com/watch?v=zwSXSltRhh0

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(filters=6, kernel_size=5, padding='same', activation='sigmoid'),
    tf.keras.layers.AveragePooling2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(filters=16, kernel_size=5, padding='valid', activation='sigmoid'),
    tf.keras.layers.AveragePooling2D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=120, activation='sigmoid'),
    tf.keras.layers.Dense(units=84, activation='sigmoid'),
    tf.keras.layers.Dense(units=10, activation='softmax')
])
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(10)
# ])

predictions = model(x_train[:1]).numpy()

tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)

# probability_model = tf.keras.Sequential([
#     model,
#     tf.keras.layers.Softmax()
# ])
#
# probability_model(x_test[:5])