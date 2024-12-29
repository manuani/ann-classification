import pickle

'''
December 28th 2024:  Another yesr coming to an end. I am determined to complete the course.
Need to be more disciplined to complete the learning. Trusting the process ignoring challenges.

In annExperiemnt.py  we read the file, did gener and geo coding, split the train and test data. Performed a
scalar operation and then dumped the encoded data into files using pickle.

In this piece of code I am starting from reading these files and strat the next process
'''

with open('/Users/muralis/AIML/data/scalerdump.pk1', 'rb') as file:
    scaler= pickle.load(file)
file.close

with open('/Users/muralis/AIML/data/geoencoder.pk1', 'rb') as file:
    geo_encoder= pickle.load(file)
file.close

with open('/Users/muralis/AIML/data/genderencoder.pk1', 'rb') as file:
    gener_encoder= pickle.load(file)
file.close

## you need to load the model now. You need load_model library from keras.model

import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model("/Users/muralis/AIML/data/muralitest.keras")

## now define the input data set

input_data = {
    'CreditScore': 800,
    'Geography': 'France',
    'Gender': 'Female',
    'Age': 60,
    'Tenure': 1,
    'Balance': 600,
    'NumOfProducts': 0,
    'HasCrCard': 0,
    'IsActiveMember': 0,
    'EstimatedSalary': 500000
}

print(input_data.keys())

## use panda to convert the set to a data frame
import pandas as pd

input_data_df = pd.DataFrame([input_data], columns=input_data.keys())
print(input_data_df.head())

##use the gender encoder to convert the data
input_data_df["Gender"] = gener_encoder.transform([input_data["Gender"]])

## Use the Geo encoder to get the geo transformation
geo_encode_vals = geo_encoder.transform([input_data_df["Geography"]])
geo_encoded_df = pd.DataFrame(geo_encode_vals.toarray(),columns=geo_encoder.get_feature_names_out())
print(geo_encoded_df.head())

## drop the Georphay column from the data set and add the one hot encoded matrix

input_data_df = pd.concat([input_data_df.drop(["Geography"],axis=1),geo_encoded_df], axis=1)

##Apply the scaler function

scaled_input_data = scaler.transform(input_data_df)
print(scaled_input_data)

## now predit the outcome
prediction = model.predict(scaled_input_data)
print(f"The prediction is {prediction}")

if(prediction[0][0] <0.5):
    print("The probability of customer leaving is very low")
else:
    print("The probability of customer leaving is very high")

