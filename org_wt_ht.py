import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

data = pd.read_csv('wt_ht-220.csv')

#Machine Learning

#Female Prediction

### Female Weight prediction

#replaceing the data from data variable to df
df = pd.DataFrame(data)

X1 = df.iloc[:,0:1]  #feactuers
Y1 = df.iloc[:,1]  #labels
#X_train.shape
#Y_train.shape
#X=Gender & height
#Y=weight

#Split the train and test data
from sklearn.model_selection import train_test_split
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1,Y1,test_size = 1/3, random_state=1)

#Training the with linear Regression
from sklearn.linear_model import LinearRegression
fe_wt=LinearRegression()
fe_wt.fit(X1_train,Y1_train)

##Prediciting results
fe_wt_pred=fe_wt.predict(X1_train)
fe_wt.predict([[167]])

#saving model with help pickle
import pickle
pickle.dump(fe_wt,open("female_wt_pred.pkl","wb"))



##Female Height Prediction

X2 = df.iloc[:,-2:-1]  #feactuers
Y2 = df.iloc[:,0] 
#X2
#Y2
#Female
#X= Female weight
#Y= Felmale height

#Split the train and test data
from sklearn.model_selection import train_test_split
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2,Y2,test_size = 1/3, random_state=1)

#Training the with linear Regression
from sklearn.linear_model import LinearRegression
fe_ht=LinearRegression()
fe_ht.fit(X2_train,Y2_train)

#Predict value in train set
fe_ht_pred=fe_ht.predict(X2_train)
fe_ht.predict([[59]])

#saving model with pickle
import pickle
pickle.dump(fe_ht,open("female_ht_pred.pkl","wb"))


#Male Prediction

## Male Weight Prediction
X3 = df.iloc[:,0:1]  #feactuers
Y3 = df.iloc[:,2]  #labels
#X=Gender & height
#Y=weight
#X3
#Y3
#Split the train and test data
from sklearn.model_selection import train_test_split
X3_train, X3_test, Y3_train, Y3_test = train_test_split(X3,Y3,test_size = 1/3, random_state=1)

#Training the with linear Regression
from sklearn.linear_model import LinearRegression
ma_wt=LinearRegression()
ma_wt.fit(X3_train,Y3_train)

#Split the train and test data
from sklearn.model_selection import train_test_split
X3_train, X3_test, Y3_train, Y3_test = train_test_split(X3,Y3,test_size = 1/3, random_state=1)

#Training the with linear Regression
from sklearn.linear_model import LinearRegression
ma_wt=LinearRegression()
ma_wt.fit(X3_train,Y3_train)

#saving model with help pickle
import pickle
pickle.dump(ma_wt,open("male_wt_pred.pkl","wb"))

##Male Height Prediction

X4 = df.iloc[:,2:3]  #feactuers
Y4 = df.iloc[:,0]  #labels
#X=Male weight
#Y=height
#X4
#Y4

#Male
#X= Male weight
#Y= Male height

#Split the train and test data
from sklearn.model_selection import train_test_split
X4_train, X4_test, Y4_train, Y4_test = train_test_split(X4,Y4,test_size = 1/3, random_state=1)

#Training the with linear Regression
from sklearn.linear_model import LinearRegression
ma_ht=LinearRegression()
ma_ht.fit(X4_train,Y4_train)

#Predicting the train set
ma_ht_pred=ma_ht.predict(X2_train)
ma_ht.predict([[59]])

#saving model with help pickle
import pickle
pickle.dump(ma_ht,open("male_ht_pred.pkl","wb"))