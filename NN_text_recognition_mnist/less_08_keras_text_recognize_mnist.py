"""Learning a four-layer fully connected model
of handwriting recognition from a database 'mnist'.
Saves result of learning into file (directory)."""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.datasets import mnist
from keras.layers import Flatten
from keras.layers import Dense

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

model = keras.Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

fit_results = model.fit(x_train, y_train_cat, batch_size=256, epochs=20, validation_split=0.05)

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

model.save('model_01')

n = 0
x = np.expand_dims(x_test[n], axis=0)
res = model.predict(x)
print(res)
print(f"Number is: {np.argmax(res)}")
plt.imshow(x_test[n], cmap=plt.cm.binary)
plt.show()

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
