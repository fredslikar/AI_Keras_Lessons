"""Uses Conv2D and an AI model for better learning
of recognition of handwritten digits (images)."""


import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.datasets import mnist
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D


"""Preparation of input and verification data..."""

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

"""Directly the model itself with the use of conv2d layers"""

model = keras.Sequential([
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    Conv2D(32, (2, 2), padding='same', activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2), strides=2),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    Conv2D(64, (2, 2), padding='same', activation='relu'),
    MaxPooling2D((2, 2), strides=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model training
fit_results = model.fit(x_train, y_train_cat, batch_size=128, epochs=10, validation_split=0.2)

# Showing results of training and validation
plt.title("Losses train/validation")
plt.plot(fit_results.history['loss'], label="Train")
plt.plot(fit_results.history['val_loss'], label="Validation")
plt.legend()
plt.grid(True)
plt.show()

plt.title("Accuracies train/validation")
plt.plot(fit_results.history['accuracy'], label="Train")
plt.plot(fit_results.history['val_accuracy'], label="Validation")
plt.legend()
plt.grid(True)
plt.show()

model.save('model_03')


# Evaluation of the quality of number recognition by a training model
# Trying to recognize the first digit
n = 0
x = np.expand_dims(x_test[n], axis=0)
res = model.predict(x)
print(res)
print(f"Number is: {np.argmax(res)}")
plt.imshow(x_test[n], cmap=plt.cm.binary)
plt.show()


# Evaluation of the recognition efficiency of the entire test sample,
# comparison of false results.
pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)
print(pred.shape)
print(pred[:20])
print(y_test[:20])

mask = pred == y_test
print(mask[:10])
x_false = x_test[~mask]
y_false = pred[~mask]
y_true = y_test[~mask]
print(x_false.shape)
print(y_false.shape)

for i in range(10):
    print('Number of NN is  ' + str(y_false[i]) + ', but real number is  ' + str(y_true[i]))
    plt.imshow(x_false[i], cmap=plt.cm.binary)
    plt.show()






# show first 25 figures from training set
# plt.figure(figsize=(28, 28))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(x_train[i], cmap=plt.cm.binary)
#
# plt.show()
