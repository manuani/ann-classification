import pandas as pd
## Read the data file 
data = pd.read_csv("/Users/muralis/AIML/data/Churn_Modelling.csv")
##print(data.head())

data=data.drop(["RowNumber","CustomerId","Surname"], axis=1)


##  I am using the label encoder from sklearn here. It numbers the values 1,2,3,4 
## This is not a good way to do it for columns having may values like city or color etc. Because it as you assign numbers 1,2,3,4, etc to colors color4 will have higher weightage that you do not want
## We will use binaryencoder from sklear or method encoder for those.
from sklearn import preprocessing
gender_encoder  = preprocessing.LabelEncoder()
data['Gender'] = gender_encoder.fit_transform(data['Gender'])
##print(data.head())

## for geography we cannot use the simple label encoder as explained above. Hence I will be using the onehotencoder. 

from sklearn.preprocessing import OneHotEncoder
geo_hot_encoder = OneHotEncoder()

geo_encoder1 = geo_hot_encoder.fit_transform(data[['Geography']])
##print(geo_encoder1.toarray())
## now build a df with encoded values
geo_encoder1_df = pd.DataFrame(geo_encoder1.toarray(), columns=geo_hot_encoder.get_feature_names_out())
##print(geo_encoder1_df.head())

## Now drop the Geography from the original data set , concatenate it with geo_encoder1 

encoded_data_df = pd.concat([data.drop(['Geography'], axis=1), geo_encoder1_df], axis=1)
##print(encoded_data_df.head())

## model selection is the library in which you have all function related to machine learning models and also function to split and train

from sklearn.model_selection  import train_test_split

y_data = encoded_data_df['Exited']
x_data = encoded_data_df.drop(['Exited'], axis=1)

x_data_train, x_data_test, y_data_train, y_data_test  = train_test_split(x_data, y_data, train_size=0.33,shuffle=True, random_state=41)

## Standardize the data train and test data with standard scalar function. new value = (old_value - Mean)/std.dev
## this way it will help the models to perform better with a wide range of values
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(with_mean=True, with_std=True)
x_data_test = scaler.fit_transform(x_data_test)
x_data_train = scaler.fit_transform(x_data_train)


'''
The "pickle" library in Python is used to serialize and deserialize Python objects, meaning it converts complex data structures like lists, dictionaries, and custom classes 
into a byte stream that can be stored in a file or transmitted over a network, and later reconstructed back into the original object when needed; essentially,
 it allows you to "save" the state of a Python object to be used later in the same program or another program. 
'''
import pickle
##Save the scaler file

with open('/Users/muralis/AIML/data/scalerdump.pk1', 'wb') as file:
    pickle.dump(scaler,file)
file.close

with open('/Users/muralis/AIML/data/geoencoder.pk1', 'wb') as file:
    pickle.dump(geo_hot_encoder,file)
file.close

with open('/Users/muralis/AIML/data/genderencoder.pk1', 'wb') as file:
    pickle.dump(gender_encoder,file)
file.close

'''
Date: 28th December 2024
Now I am going to try and write the code to train the ann model
The work involved will be
1. Define the model (Sequential) -- model needs Dense --> activation, nodes, shape 
2. Compile the model --> optimizer, loss
3. Define callback
4. Define early stop
5. train the model -- model.fit
6. Save the model
7. Open the tensorboard and review
'''

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input

model = Sequential([
    Input(shape=(x_data_train.shape[1],)),
    Dense(64,activation='relu'),
    Dense(32,activation='relu'),
    Dense(1,activation='sigmoid')
    ]
)

print(model.summary())

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

opt = Adam(learning_rate = 0.01)
los= BinaryCrossentropy()

## now that we have defined the optimizer and the loss we can call the compile 

model.compile(opt,los,metrics =['accuracy'])

## Now let us define the callback earlystopping and Tensorboard that will be used in the fitment

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

est = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

##define directory to use the logfies
import datetime
log_dir = '/Users/muralis/AIML/data/log/fit/' +datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tb = TensorBoard(log_dir=log_dir, histogram_freq=1)

print(x_data_train.shape)
print(y_data_train.shape)
print(x_data_test.shape)
print(y_data_test.shape)

hist = model.fit(x=x_data_train, y=y_data_train, epochs=150,callbacks=[est,tb],validation_data=(x_data_test,y_data_test) )

from tensorflow.keras.saving import save_model
save_model(model,'/Users/muralis/AIML/data/muralitest.keras')

# Launch TensorBoard
import subprocess
subprocess.Popen(["tensorboard", "--logdir", log_dir, "--port", "6006"])

# The script will continue running while TensorBoard runs on port 6006
print("TensorBoard is running on http://localhost:6006/")


