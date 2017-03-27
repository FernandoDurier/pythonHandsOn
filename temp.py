# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Import the libraries 
import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;


dataset = pd.read_csv('Data.csv');

x = dataset.iloc[:, :-1].values;
y = dataset.iloc[:,3].values;

#Taking Care of Missing Data

from sklearn.preprocessing import Imputer

#first param is the missing value the second is the strategy to replace the Not a Number values
#the third param is to where apply the strategy using axis =0 apply the strategy through the columns
#and if axis = 1, it will apply the strategy to the row;

imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)

#.fit is used to take the data from imputer insert into x but not in all collumns
#that is why we need to use x[:,1:3] that means -> : -all rows and 1:3 -from collumn 1 to 2, the greater bound is excluded

imputer = imputer.fit(x[:, 1:3])

#finally here we have the replacement using the method transform 
#here we have the nan values replaced by the mean of all sallaries

x[:, 1:3] = imputer.transform(x[:, 1:3])

#there are other possibilities for strategy param in imputer
#the options are mean, median and most_frequent


#pre process categorical data using LabelEncoder

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_x = LabelEncoder()
#now we can transform the categorical values of the matrix in encoded values
# here we take all rows in the matrix and the column 0
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])

#for not make machine learning algorithm think that the countries 
#have any ordinal relationship and eventually confound the models then we need dummy 
#encoding that is basically "normalization" of the countries

onehotencoder = OneHotEncoder(categorical_features=[0])
x = onehotencoder.fit_transform(x).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


#spliting dataset into test and training 
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.2, random_state)
