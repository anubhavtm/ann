
"""
Created on Thu Jul  4 19:49:37 2019

@author: anubh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Churn_Modelling.csv');

x=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x_1=LabelEncoder()
x[:,1]=labelencoder_x_1.fit_transform(x[:,1])

labelencoder_x_2=LabelEncoder()
x[:,2]=labelencoder_x_2.fit_transform(x[:,2])


"one hot encoding"

onehotencoder=OneHotEncoder(categorical_features=[1])
x=onehotencoder.fit_transform(x).toarray()

x=x[:,1:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_split=train_test_split(x,y,test_size=0.2,random_state=0)

"feature scaling"
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)



from keras.models import Sequential
from keras.layers import Dense
classifier= Sequential()
"first hidden layer"
classifier.add(Dense(output_dim=6,input_dim=11,init='uniform',activation='relu'))
"second hidden layer"

classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
"output layer"
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))


"compiling model"
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

"run the model"
classifier.fit(x_train,y_train,batch_size=10,epochs=100)

"test model"

y_pred=classifier.predict(x_test)
y_pred=(y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_split,y_pred)

"manual test"
"""
Geography:France
Credit Score:600
Gender Male
Age:40
Tenure:3
Balance:6000
No of prod:2
has credit c:yes
is active mamber:yes
estimated salary:50000
"""
x_m=np.array([[0,0,600,1,40,3,6000,2,1,1,50000]])
x_m=sc.fit_transform(x_m)
y_m=classifier.predict(x_m)
y_m=y_m > 0.5



from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier= Sequential()
    classifier.add(Dense(output_dim=6,input_dim=11,init='uniform',activation='relu'))
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
    classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier

classifier=KerasClassifier(build_fn=build_classifier,batch_size=10,epochs=100)
accuracy=cross_val_score(estimator=classifier,X=x_train,y=y_train,cv=10,n_jobs=-1)
