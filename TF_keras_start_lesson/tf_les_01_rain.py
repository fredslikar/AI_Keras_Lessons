"""Learning to imitate a person: the probability of rain
and the probability that people take an umbrella"""

import keras as k
import numpy as np
from keras.layers import Dense


input_data = np.array([0.0, 0.3, 0.7, 0.9, 1.0])
output_data = np.array([0.0, 0.5, 0.9, 1.0, 1.0])

"""<input_shape=(1,)> - we set the number of features (we look at the bit depth of the model) 
we have a vector [0.3, 0.7, 0.9] its bit is one, so we specify (1,) as a tuple in <input_shape=(1,)>"""

modelk = k.Sequential()
# modelk.add(k.layers.Dense(units=1, activation="linear"))
modelk.add(k.layers.Dense(units=1, input_shape=(1,), activation="linear"))
modelk.compile(loss="mse", optimizer="sgd")
fit_result = modelk.fit(input_data, output_data, epochs=100)

# Testing model
predicted = modelk.predict([0.5])
print(predicted)
