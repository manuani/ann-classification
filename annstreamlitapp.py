import pickle
import pandas as pd
import numpy as np
import tensorflow
import streamlit as st

from tensorflow.keras.models import load_model

with open('/Users/muralis/AIML/data/scalerdump.pk1', 'rb') as file:
    scaler= pickle.load(file)

with open('/Users/muralis/AIML/data/geoencoder.pk1', 'rb') as file:
    geo_encoder= pickle.load(file)

with open('/Users/muralis/AIML/data/genderencoder.pk1', 'rb') as file:
    gener_encoder= pickle.load(file)

## you need to load the model now. You need load_model library from keras.model
model = load_model("/Users/muralis/AIML/data/muralitest.keras")


credit_score=st.slider('CreditScore',300,900)
geography=st.selectbox('Geography',geo_encoder.categories_[0])
gender=st.selectbox('Gender',gener_encoder.classes_)
age=st.slider('Age',18,100)
tenure=st.slider('Tenure',1,10)
balance=st.number_input('Balance')
numberofproducts= st.slider('NumOfProducts',1,4)
has_cr_cared = st.checkbox('Has Credit Card',[0,1])
is_active = st.checkbox('Is Active Member',[0,1])
estimated_salary = st.number_input("Estimated Salary")

input_data_df = pd.DataFrame(
    {
        "CreditScore":[credit_score],
        "Geography":[geography],
        "Gender":[gender],
        "Age":[age],
        "Tenure":[tenure],
        "Balance":[balance],
        "NumOfProducts":[numberofproducts],
        "HasCrCard":[has_cr_cared],
        "IsActiveMember":[is_active],
        "EstimatedSalary":[estimated_salary]
    }
)

input_data_df["Gender"] = gener_encoder.transform([input_data_df["Gender"]])

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
st.write(f"The prediction is {prediction}")

if(prediction[0][0] <0.5):
    st.write("The probability of customer leaving is very low")
else:
    st.write("The probability of customer leaving is very high")