import tensorflow as tf

"""We create a fully connected class model by inheriting from the tensorflow module module class;
outputs - the number of neurons at the output;
fl_init - flag value."""


class DenseNN(tf.Module):
    def __init__(self, outputs):
        super().__init__()  # base class constructor
        self.outputs = outputs
        self.fl_init = False

    def __call__(self, x):  # it is supposed to take a matrix as input
        if not self.fl_init:
            self.w = tf.random.truncated_normal((x.shape[-1], self.outputs), stddev=0.1, name='w')
            self.b = tf.ones(self.outputs, name='b')

            self.w = tf.Variable(self.w)
            self.b = tf.Variable(self.b)

            self.fl_init = True

        y = x @ self.w + self.b
        return y


model = DenseNN(1)

"""Determine the set of training sample: 100 by 2
- minval is the minimum value
- maxval is the maximum value in a random sample
- shape 100 by 2 - means that in the end a matrix of 100 rows of 2 values should be returned
(2 columns each (x1 and x2)"""

x_train = tf.random.uniform(minval=0, maxval=10, shape=(100, 2))

"""Determine the set of correct answers for training.
Each answer will be equal to the sum of the first and second value for each line"""

y_train = [a + b for a, b in x_train]

"""Determine the loss function - loss - this will be the standard deviation"""

loss = lambda y_fact, y_model: tf.reduce_mean(tf.square(y_fact - y_model))

"""We define the optimizer adam with a learning step of 0.01 
(the optimizer using the data of the loss function
changes subsequent parameters (weighting coefficients of 
the function so as to approach the real value
by multiple correction, using the loss error function)"""

opt = tf.optimizers.Adam(learning_rate=0.01)

"""Let's write the direct training model of our model as follows:
- number of epochs 50;
- transform x into a matrix by adding the zero axis (first) to the dimension of the x tensor,
    after which it with 2 columns becomes 1 row - 2 columns;
- for each epoch we will sort through the entire training set
  and on this set, respectively, determine the gradients and determine the parameters w and b"""

EPOCHS = 50
for n in range(EPOCHS):
    for x, y in zip(x_train, y_train):
        x = tf.expand_dims(x, axis=0)  # converts the inputs to a matrix
        y = tf.constant(y, shape=(1, 1))

        with tf.GradientTape() as tape:
            f_loss = loss(y, model(x))  # loss calculations using the loss function

        # calculate f_loss derivatives with respect to b and w
        # optimize the values of the variables b and w
        # according to the previously specified ADAM optimizer
        grads = tape.gradient(f_loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

    print(f_loss.numpy())

print(model.trainable_variables)
