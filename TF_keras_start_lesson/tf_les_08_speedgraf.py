"""This uses <@tf.function> to speed up calculations.
The speed can be compared with the code from lesson 07."""

"""Connect the OS so that there are no memory failures"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from keras.datasets import mnist
from keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

x_train = tf.reshape(tf.cast(x_train, tf.float32), [-1, 28 * 28])
x_test = tf.reshape(tf.cast(x_test, tf.float32), [-1, 28 * 28])

y_train = to_categorical(y_train, 10)


class DenseNN(tf.Module):
    def __init__(self, outputs, activate='relu'):
        super().__init__()
        self.outputs = outputs
        self.activate = activate
        self.fl_init = False

    def __call__(self, x):
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
TOTAL_SIZE = x_train.shape[0]

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)


@tf.function
def train_batch(x_batch, y_batch):
    with tf.GradientTape() as tape:
        f_loss = cross_entropy(y_batch, model_predict(x_batch))

    grads = tape.gradient(f_loss, [layer_1.trainable_variables, layer_2.trainable_variables])
    opt.apply_gradients(zip(grads[0], layer_1.trainable_variables))
    opt.apply_gradients(zip(grads[1], layer_2.trainable_variables))
    return f_loss


for n in range(EPOCHS):
    loss = 0
    for x_batch, y_batch in train_dataset:
        loss = loss + train_batch(x_batch, y_batch)
    print(loss.numpy())

y = model_predict(x_test)
y2 = tf.argmax(y, axis=1).numpy()
acc = len(y_test[y_test == y2]) / y_test.shape[0] * 100
print(acc)
