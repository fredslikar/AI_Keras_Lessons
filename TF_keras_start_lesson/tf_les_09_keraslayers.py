import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from keras.datasets import mnist
from keras.utils import to_categorical


class DenseLayer(tf.keras.layers.Layer):

    def __init__(self, units=1):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        """Dimensions of the weight matrix;
        filling with normal distribution method;
        trainable variables."""

        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer='zeros', trainable=True)

    def call(self, inputs):
        """We can create an additional error calculation, it will also be minimized when training nn
        and add this error to the main error calculations;
        multiply the input by the weights of the double matrix and add the offset b
        when multiplying, the rule of multiplication of matrices column1 = row 2"""

        regular_loss = tf.reduce_mean(tf.square(
            self.w))
        self.add_loss(regular_loss)
        return tf.matmul(inputs, self.w) + self.b


class NeuralNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.layer1 = DenseLayer(128)
        self.layer2 = DenseLayer(10)

    def call(self, inputs):
        x = self.layer1(inputs)
        x = tf.nn.relu(x)
        x = self.layer2(x)
        x = tf.nn.softmax(x)
        return x


model = NeuralNetwork()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

x_train = tf.reshape(tf.cast(x_train, tf.float32), [-1, 28 * 28])
x_test = tf.reshape(tf.cast(x_test, tf.float32), [-1, 28 * 28])

y_train = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

model.fit(x_train, y_train, batch_size=32, epochs=5)

print(model.evaluate(x_test, y_test_cat))
