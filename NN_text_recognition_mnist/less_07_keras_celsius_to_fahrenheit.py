import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Dense

c = np.array([-40, -10, 0, 8, 15, 22, 38])
f = np.array([-40, 14, 32, 46, 59, 72, 100])

model = keras.Sequential()
model.add(Dense(units=2, input_shape=(1,), activation='linear'))
model.add(Dense(units=1, input_shape=(2,), activation='linear'))

model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.1))
history = model.fit(c, f, epochs=250, verbose=1)

print(model.predict([100, 150]))
print(model.get_weights())

plt.plot(history.history['loss'])
plt.grid(True)
plt.show()
