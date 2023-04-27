"""Connect the OS so that there are no memory failures"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

"""Import a database of images 28x28 handwritten numbers"""

from keras.datasets import mnist

"""We import a method that allows you to translate the resulting data into 
a categorical form, for example:
5 to 0000010000; 0 to 1000000000; 1 to 0100000000 etc."""

from keras.utils import to_categorical

"""We load the data into the corresponding variables (x_train and y_train, x_test and y_test) from
mnist database (images 28x28 handwritten digits).
Images 28x28=784 pixels are loaded into x_train, each of which has a value from
0 to 255. Use the following code: """

(x_train, y_train), (x_test, y_test) = mnist.load_data()

"""We bring the data into the range from 0 to 1 by 
dividing the current value by the maximum possible"""

x_train = x_train / 255
x_test = x_test / 255

"""Convert the data first to floating point data using <cast float32>,
further -1 means such a proportional value of the number of rows of the matrix 
(in our case, instead of the matrix, the new data will be have the form of a vector, 
i.e. only 784 columns instead of 28*28 and in the amount of 60000 rows)
proportional to its original dimension 28*28, i.e. -1 will return the value 60000.
That is, we transform the data tensor from 60000 * 28 * 28 into a matrix of 60000 * 784 """

x_train = tf.reshape(tf.cast(x_train, tf.float32), [-1, 28 * 28])
x_test = tf.reshape(tf.cast(x_test, tf.float32), [-1, 28 * 28])

"""We translate the resulting data Y into a categorical form, for example:
5 to 0000010000; 0 to 1000000000; 1 to 0100000000 etc."""

y_train = to_categorical(y_train, 10)
# y_test = to_categorical(y_test, 10)

"""We use the model from lesson 06, 
just add the form of the <relu> activation function"""


class DenseNN(tf.Module):
    def __init__(self, outputs, activate='relu'):  # add the type of <relu> activation function
        super().__init__()
        self.outputs = outputs
        self.activate = activate  # add the type of <relu> activation function
        self.fl_init = False

    def __call__(self, x):  # it is supposed to take a matrix as input
        if not self.fl_init:
            self.w = tf.random.truncated_normal((x.shape[-1], self.outputs), stddev=0.1, name='w')
            self.b = tf.ones(self.outputs, name='b')

            self.w = tf.Variable(self.w)
            self.b = tf.Variable(self.b, trainable=False)

            self.fl_init = True

        y = x @ self.w + self.b

        if self.activate == 'relu':
            return tf.nn.relu(y)
        elif self.activate == 'softmax':
            return tf.nn.softmax(y)

        return y


layer_1 = DenseNN(128)
layer_2 = DenseNN(10)


def model_predict(x):
    y = layer_1(x)
    y = layer_2(y)
    return y


cross_entropy = lambda y_true, y_pred: tf.reduce_mean(tf.losses.categorical_crossentropy(y_true, y_pred))
opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)

BATCH_SIZE = 32
EPOCHS = 10
TOTAL_SIZE = x_train.shape[0]  # 60000 size

print('11111111111111111111111111111111111111111111111')
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))  # create slices (picture - value)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)  # shuffle slices and form batches
for n in range(EPOCHS):
    loss = 0
    for x_batch, y_batch in train_dataset:
        with tf.GradientTape() as tape:
            f_loss = cross_entropy(y_batch, model_predict(x_batch))

        loss = loss + f_loss
        grads = tape.gradient(f_loss,
                              [layer_1.trainable_variables, layer_2.trainable_variables])  # calculate derivatives
        opt.apply_gradients(zip(grads[0], layer_1.trainable_variables))
        opt.apply_gradients(zip(grads[1], layer_2.trainable_variables))  # using derivatives optimize variables
    print(loss.numpy())

"""Check the robotic ability of the neuron using the following code:
- comb out all;
- y2 will be equal to the index of the maximum argument in the line (the maximum is determined line by line),
there will be one big column"""

y = model_predict(x_test)
y2 = tf.argmax(y, axis=1).numpy()
acc = len(y_test[y_test == y2]) / y_test.shape[0] * 100
print(acc)
