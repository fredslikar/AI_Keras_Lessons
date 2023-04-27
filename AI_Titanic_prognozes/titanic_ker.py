import keras as k
import pandas as pd
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import openpyxl

"""Neural network - prognosis of chances of survives with use 'Titanic's passengers data:
age, gender, class."""

"""Save into variable data_frame file.csv via pandas;
select the desired data columns for the input data;
select the desired data columns for the OUTPUT data."""

data_frame = pd.read_csv('titanic.csv')
input_names = ['Age', 'Sex', 'Pclass']
output_names = ['Survived']

"""To convert data by age in the range from 0 to 1 - enter the maximum age.
Divide the actual current age by the maximum and enter the range from 0 to 1;
0 - female, 1 - male;
classes - change to 1x3 vector;"""

max_age = 100
encoders = {"Age": lambda age: [age / max_age],
            "Sex": lambda gen: {"male": [0], "female": [1]}.get(gen),
            "Pclass": lambda pclass: {1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 1]}.get(pclass),
            "Survived": lambda s_value: [s_value]}


def dataframe_to_dict(df):
    """Move all data from a pandasfile (csv) to a dictionary.
    For every column in pandasfile variable take its values.
    Key - name of column - (str), values - values of this column -(array([])).
    Result is dictionary."""

    result = dict()
    for column in df.columns:
        values = data_frame[column].values
        result[column] = values
    return result


def make_supervised(df):
    """Create a dictionary with two keys:
    1 - input data; 2 - output data (actual result).
    Pre-save the input (three values:) and output (survive) data in the appropriate variables.
    Use 'dataframe_to_dict' func. for format data for dictionary"""

    raw_input_data = data_frame[input_names]
    raw_output_data = data_frame[output_names]
    return {"inputs": dataframe_to_dict(raw_input_data),
            "outputs": dataframe_to_dict(raw_output_data)}


def encode(data):
    """Encoding data for use this data with Keras (the data must be a dictionary).
    1 - use pairwise iteration.
    2 - run each value from the dictionary through lambda functions in 'encoders'-dictionary.
    3 - then add encoded values to the list 'vector'.
    As a result, we get a list of lists of all input values element by element"""

    vectors = []
    for data_name, data_values in data.items():
        encoded = list(map(encoders[data_name], data_values))
        vectors.append(encoded)
    formatted = []
    for vector_row in list(zip(*vectors)):
        vector = []
        for element in vector_row:
            for e in element:
                vector.append(e)
        formatted.append(vector)
    return formatted


"""Get input and output data in the required format.
Form a training set and a test set."""

supervised = make_supervised(data_frame)
encoded_inputs = np.array(encode(supervised['inputs']))
encoded_outputs = np.array(encode(supervised['outputs']))

train_x = encoded_inputs[:600]
train_y = encoded_outputs[:600]

test_x = encoded_inputs[600:]
test_y = encoded_outputs[600:]

"""Create the first layer of the model with 5 inputs and 5 outputs.
Create the second layer of the model from 5 inputs and 1 output.
Compile the model."""

modelk = k.Sequential()
modelk.add(k.layers.Dense(units=5, input_shape=(5,), activation="relu"))
modelk.add(k.layers.Dense(units=1, activation="sigmoid"))
modelk.compile(loss="mse", optimizer="sgd", metrics=['accuracy'])

"""Model training..."""

fit_results = modelk.fit(x=train_x, y=train_y, epochs=10, validation_split=0.25)

"""Visual display of model training...
Loss & accuracy..."""

plt.title("Losses train/validation")
plt.plot(fit_results.history['loss'], label="Train")
plt.plot(fit_results.history['val_loss'], label="Validation")
plt.legend()
plt.show()

plt.title("Accuracies train/validation")
plt.plot(fit_results.history['accuracy'], label="Train")
plt.plot(fit_results.history['val_accuracy'], label="Validation")
plt.legend()
plt.show()


"""The work of the model on the test sample. 
The real work of the model. 
Display of weight coefficients and results. Saving results in csv & xlsx files"""

predicted_test = modelk.predict(test_x)
real_data = data_frame.iloc[600:][input_names + output_names]
real_data["PSurvived"] = predicted_test
print(real_data)
print(type(real_data))
modelk.summary()
w = modelk.get_weights()
print(w)
print(modelk.evaluate(test_x, test_y))
real_data.to_csv(r'prognoses_titanic_files/ai_titanic.csv')
real_data.to_excel(r'prognoses_titanic_files/ai_titanic.xlsx')
