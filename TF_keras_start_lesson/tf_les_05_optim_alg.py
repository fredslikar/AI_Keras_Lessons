import matplotlib.pyplot as plt
import tensorflow as tf

TOTAL_POINTS = 1000


x = tf.random.uniform(shape=[TOTAL_POINTS], minval=0, maxval=10)
noise = tf.random.normal(shape=[TOTAL_POINTS], stddev=0.2)

k_true = 0.7
b_true = 2.0

y = x * k_true + b_true + noise

plt.scatter(x, y, s=2)
plt.show()

k = tf.Variable(0.0)
b = tf.Variable(0.0)

EPOCHS = 500
learning_rate = 0.02

BATCH_SIZE = 100
num_step = TOTAL_POINTS // BATCH_SIZE

for n in range(EPOCHS):
    with tf.GradientTape() as tape:
        f = k * x + b
        loss = tf.reduce_mean(tf.square(y - f))
    dk, db = tape.gradient(loss, [k, b])

    k.assign_sub(learning_rate * dk)
    b.assign_sub(learning_rate * db)

print(k, b, sep='\n')

y_pr = k * x + b
plt.scatter(x, y, s=2)
plt.scatter(x, y_pr, c='r', s=2)
plt.show()