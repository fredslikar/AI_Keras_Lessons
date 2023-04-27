"""Determination of the derivative with respect to x and y of a function f
with use <with tf.GradientTape() as tape:>"""

import tensorflow as tf

x = tf.Variable([[2.0]])
y = tf.Variable([[-4.0]])

with tf.GradientTape() as tape:
    f = (x+y)**2 + 2*x*y

df = tape.gradient(f, [x, y])
print(df[0], df[1], sep='\n')
