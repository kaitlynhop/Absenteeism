#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import pickle

# absenteeism class
class AbsenteeismModel():
    
#     define attributes with logistic and scaler models
    def __init__(self, model_file, scaler_file):
#         read files to objects
        with open("model", "rb") as model_file, open("scaler", "rb") as scaler_file:
            self.regressor = pickle.load(model_file)
            self.scaler = pickle.load(scaler_file)
#             initialize data attribute on obj instance
            self.data = None
    
    def load_and_clean_data(self, data_file):
#       import data
        df = pd.read_csv(data_file, delimiter=",")
#       copy raw dataframe and assign to object attribute
        self.df_with_predictions = df.copy()
#       drop ID cols 
        df.drop(columns=['ID'], inplace=True)

#       Encode Reasons columns
        encoded_reasons = pd.get_dummies(df['Reason for Absence'], drop_first = True)

#       Group Reasons
        df["Reason_1"] = encoded_reasons.loc[:,1:14].max(axis=1)
        df["Reason_2"] = encoded_reasons.loc[:,15:17].max(axis=1)
        df["Reason_3"] = encoded_reasons.loc[:,18:21].max(axis=1)
        df["Reason_4"] = encoded_reasons.loc[:,22:28].max(axis=1)
#       fill null values with 0
        df = df.fillna(value=0)

#       Month values and day of the week
        df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
        df["Month Values"] = df["Date"].apply(lambda x: x.month)
        df["Day of the Week"] = df["Date"].apply(lambda x: x.weekday())
        
#       Encode Education
        df["Education"] = df["Education"].map({1:0, 2:1, 3:1, 4:1})
    
#       Drop raw reasons and raw date and target variables
        df.drop(columns=['Reason for Absence', 'Date'], inplace = True)
    
#       Keep checkpoint of preprocessed data with ordered columsn
        df = df[['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Month Values',
       'Day of the Week', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets']]
    
        self.preprocessed_data = df.copy()
        
#       data attribute is scaled data
        self.data = np.array(self.scaler.transform(df))
    
#     function to get probability data for features
    def predicted_probability(self):
        if (self.data is not None):
            predicted_probs = self.regressor.predict_proba(self.data)[:,1]
            return predicted_probs
        
#   function to get output for features
    def predicted_output_category(self):
        if (self.data is not None):
            predicted_outputs = self.regressor.predict(self.data)
            return predicted_outputs
        
#   function to get prob and output and add to original - clean data columns
    def predicted_outputs(self):
        if (self.data is not None):
            self.preprocessed_data["Probability"] = self.regressor.predict_proba(self.data)[:,1]
            self.preprocessed_data["Prediction"] = self.regressor.predict(self.data)
            return self.preprocessed_data


        
        


# In[ ]:




