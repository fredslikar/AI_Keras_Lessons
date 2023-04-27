import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.datasets import cifar10
from keras.utils import to_categorical

tf.random.set_seed(1)

input_data = keras.Input(shape=(32, 32, 3))
x = layers.Conv2D(32, 3, activation='relu')(input_data)
x = layers.MaxPooling2D(2, padding='same')(x)
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(2, padding='same')(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
output_data = layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=input_data, outputs=output_data)
model.summary()


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train / 255
x_test = x_test / 255

y_train = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=64, epochs=2, validation_split=0.2)
print(model.evaluate(x_test, y_test_cat))
