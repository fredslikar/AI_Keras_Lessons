import tensorflow as tf
import numpy as np


# creating constants TF
a = tf.constant(1)
print(a)

b = tf.constant(1, shape=(1, 1))
print(b)

c = tf.constant([1, 2, 3, 4])
print(c)

d = tf.constant([[1, 2],
                 [3, 4],
                 [5, 6]], dtype=tf.float32)
print(d)

a2 = tf.cast(a, dtype=tf.float64)
print(a2)

e = tf.constant([[11, 12], [21, 22], [31, 32]], dtype=tf.float32)

# creating variables TF
v1 = tf.Variable(-1.2)
v2 = tf.Variable([4, 5, 6, 7])
v3 = tf.Variable(e)
print(v1, v2, v3, sep='\n\n')

# replacement of values in a variable (you need to observe the dimension of the tensor (data matrix)
v1.assign(2)
v2.assign([1, 3, 4, 7])
v3.assign(d)
print(v1, v2, v3, sep='\n\n')

# adding to each element the value corresponding to this element
v2.assign_add([1, 1, 1, 1])
print(v2)

# subtract from each element the value corresponding to this element
v2.assign_sub([2, 2, 2, 2])
print(v2)

# create a link and read the tensor at index 0
val_0 = v3[0]
val_12 = v3[1:3]
print(val_0, val_12)

# change through the variable-reference val_0 the value at index 0 of the v3 tensor element
# at the same time, the value of the variable itself, val_0, does not change
# (but the first recorded value of the tensor element remains)
val_0.assign(10)
print(v3)
print(val_0)

# changing the dimension of the tensor, from a vector
# to a 5x6 matrix (in the example below) using <reshape>:
a = tf.constant(range(30))
b = tf.reshape(a, [5, 6])
print(b)
