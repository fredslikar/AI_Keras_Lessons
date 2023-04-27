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


model = keras.models.load_model('model_01')

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