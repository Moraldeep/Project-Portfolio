
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[4]:


from sklearn import cross_validation
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


# In[339]:


from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


# In[1]:


import lightgbm as lgb


# In[287]:


titanic_train = pd.read_csv("titanic_train.csv")
titanic_test = pd.read_csv("titanic_test.csv")
titanic_train.head(10)


# In[288]:


titanic_train.shape


# In[289]:


titanic_test.shape


# In[290]:


titanic_train.describe()


# #Age has 177 missing values. replacing by group mean

# In[291]:


titanic_test.describe()


# In[292]:


titanic_train['Age'] = titanic_train.groupby(['Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
titanic_test['Age'] = titanic_test.groupby(['Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))


# In[293]:


titanic_test['Fare'].fillna(titanic_test["Fare"].median(),inplace = True)


# In[294]:


titanic_train.info()


# In[295]:


titanic_train['Embarked'].fillna(titanic_train["Embarked"].value_counts().index[0],inplace = True)


# As only 2 rows have missing values. I replaced them using mode

# In[296]:


titanic_test.info()


# #More than 75% of the values in Cabin Column is missing. So, I removed the column from the consideration

#  # Percentage of Passengers Died/Survived

# In[297]:


(titanic_train["Survived"].value_counts(normalize=True)*100).plot(kind='bar', color="red", alpha=.65)


# 61.6% people died during the titantic sink

# # Counting number of Passengers Died/Survived by Sex

# In[298]:


titanic_train["Sex"].value_counts()


# In[299]:


titanic_train[titanic_train["Survived"]==1]["Sex"].value_counts()


# In[300]:


survived = titanic_train[titanic_train["Survived"] == 1]["Sex"].value_counts()
dead = titanic_train[titanic_train["Survived"] == 0]["Sex"].value_counts()
percent_female_survived = survived["female"]/titanic_train["Sex"].value_counts()["female"]*100
percent_male_survived = survived["male"]/titanic_train["Sex"].value_counts()["male"]*100
print("Percentage female survived =",percent_female_survived,"%")
print("Percentage male survived =",percent_male_survived,"%")


# In[301]:


# Counting number of Passengers Died/Survived by Embarkment


# In[302]:


survived_e = titanic_train[titanic_train["Survived"] == 1]["Embarked"].value_counts()
percent_survived_S = survived_e[0]/titanic_train["Embarked"].value_counts()[0]*100
percent_survived_C = survived_e[1]/titanic_train["Embarked"].value_counts()[1]*100
percent_survived_Q = survived_e[2]/titanic_train["Embarked"].value_counts()[2]*100
print("Percentage survived in Embarked S =",percent_survived_S,"%")
print("Percentage survived in Embarked C =",percent_survived_C,"%")
print("Percentage survived in Embarked Q =",percent_survived_Q,"%")


# In[303]:


survived_class = titanic_train[titanic_train["Survived"] == 1]["Pclass"].value_counts(sort = False)
percent_survived_1 = survived_class.iloc[0]/titanic_train["Pclass"].value_counts(sort = False).iloc[0]*100
percent_survived_2 = survived_class.iloc[1]/titanic_train["Pclass"].value_counts(sort = False).iloc[1]*100
percent_survived_3 = survived_class.iloc[2]/titanic_train["Pclass"].value_counts(sort = False).iloc[2]*100
print("Percentage survived in Class 1 =",percent_survived_1,"%")
print("Percentage survived in Class 2 =",percent_survived_2,"%")
print("Percentage survived in Class 3 =",percent_survived_3,"%")


# In[304]:


titanic_train[titanic_train["Survived"] == 1]["SibSp"].value_counts(sort = False)
titanic_train["SibSp"].value_counts(sort = False)


# In[305]:


titanic_train[titanic_train["Survived"] == 1]["Parch"].value_counts(sort = False)
titanic_train["Parch"].value_counts(sort = False)


# # Data Preprocessing

# In[306]:


titanic_train['Title'] = titanic_train['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
titanic_test['Title'] = titanic_test['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())


# In[307]:


titanic_train.drop(['PassengerId','Cabin',"Name",'Ticket'],axis = 1,inplace = True)


# In[308]:


titanic_test.drop(['PassengerId','Cabin',"Name",'Ticket',],axis = 1,inplace = True)


# In[309]:


titanic_train['Age'].hist()


# In[310]:


titanic_train['Fare'].hist()


# In[311]:


bins_age = [0,10,20,30,50,80]
labels_age = [1,2,3,4,5]
titanic_train['Age_binned'] = pd.cut(titanic_train['Age'], bins=bins_age, labels=labels_age)


# In[312]:


bins_fare = [0,25,50,100,300,600]
labels_fare = [1,2,3,4,5]
titanic_train['Fare_binned'] = pd.cut(titanic_train['Fare'], bins=bins_fare, labels=labels_fare)


# In[313]:


titanic_train.drop(["Age","Fare"],axis=1,inplace=True)


# In[314]:


titanic_test['Age_binned'] = pd.cut(titanic_test['Age'], bins=bins_age, labels=labels_age)
titanic_test['Fare_binned'] = pd.cut(titanic_test['Fare'], bins=bins_fare, labels=labels_fare)
titanic_test.drop(["Age","Fare"],axis=1,inplace=True)


# In[315]:


titanic_train["Family_Size"] = titanic_train["SibSp"]+titanic_train["Parch"]
titanic_test["Family_Size"] = titanic_test["SibSp"]+titanic_test["Parch"]


# In[316]:


titanic_train["Family_Size"].hist()


# In[317]:


bins_family = [-1,1,4,10]
labels_family = ['Single','Medium_Family','Large_Family']
titanic_train['Family_Size_Binned'] = pd.cut(titanic_train['Family_Size'], bins=bins_family, labels=labels_family)
titanic_test['Family_Size_Binned'] = pd.cut(titanic_test['Family_Size'], bins=bins_family, labels=labels_family)


# In[318]:


titanic_train.drop(["SibSp","Parch","Family_Size"],axis=1,inplace=True)
titanic_test.drop(["SibSp","Parch","Family_Size"],axis=1,inplace=True)


# In[319]:


x_train = titanic_train.drop(['Survived'],axis = 1)
combined = x_train.append(titanic_test)
combined.reset_index(inplace=True)
combined.drop('index', inplace=True, axis=1)


# In[320]:


y_train = titanic_train['Survived']


# In[321]:


combined_n = pd.get_dummies(combined,columns= ["Pclass","Sex","Embarked","Title",'Age_binned','Fare_binned','Family_Size_Binned'],drop_first=True);


# In[322]:


combined_n.shape


# In[323]:


titanic_train_n= combined_n.iloc[0:891,:]
titanic_test_n = combined_n.iloc[891:1309,:]


# In[324]:


titanic_test_n.tail(1)


# # Dividing the training set into training and validation set

# In[332]:


train, validation,y_tr,y_val = cross_validation.train_test_split(titanic_train_n,y_train,test_size = 0,random_state = 0)


# # Model Fit

# In[326]:


classifier_LR = LogisticRegression(penalty='l2',random_state = 0)
classifier_RF = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier_GB = GaussianNB()
classifier_SVC = SVC(kernel = 'rbf', random_state = 0)
classifier_KNN = KNeighborsClassifier(n_neighbors = 9, metric = 'minkowski', p = 2)


# In[333]:


accuracies_LR = cross_val_score(estimator = classifier_LR, X=train , y=y_tr , cv = 10)
accuracies_RF = cross_val_score(estimator = classifier_RF, X=train , y=y_tr , cv = 10)
accuracies_GB = cross_val_score(estimator = classifier_GB, X=train , y=y_tr , cv = 10)
accuracies_SVC = cross_val_score(estimator = classifier_SVC, X=train , y=y_tr , cv = 10)
accuracies_KNN = cross_val_score(estimator = classifier_KNN, X=train , y=y_tr , cv = 10)


# In[334]:


print("Accuracy - Logistic Regression:", accuracies_LR.mean())
print("Accuracy - Random Forest:", accuracies_RF.mean())
print("Accuracy - Naive Gaussian:", accuracies_GB.mean())
print("Accuracy - Support_Vector:", accuracies_SVC.mean())
print("Accuracy - K-Nearest Neighbor:", accuracies_KNN.mean())


# In[336]:


classifier_KNN.fit(train,y_tr)
classifier_RF.fit(train,y_tr)
classifier_LR.fit(train,y_tr)
classifier_SVC.fit(train,y_tr)


# In[330]:


y_predict_KNN = classifier_KNN.predict(validation)
y_predict_RF = classifier_RF.predict(validation)
y_predict_LR = classifier_LR.predict(validation)
y_predict_SVC = classifier_LR.predict(validation)


# In[331]:


print("Logistic Regression-Prediction on Validation-Set",(y_predict_LR == y_validation).mean())
print("Random Forest-Prediction on Validation-Set",(y_predict_RF == y_validation).mean())
print("K Nearest Neighbour-Prediction on Validation-Set",(y_predict_KNN == y_validation).mean())


# # Tuning Parameters 

# In[340]:


parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}


# In[343]:


svc = SVC()


# In[344]:


clf = GridSearchCV(svc, parameters)


# # For Prediction on Test Set

# In[207]:


x = titanic_train_n
y = y_train


# In[211]:


classifier_LR = LogisticRegression(penalty='l2',random_state = 0)


# In[212]:


classifier_LR.fit(x,y)


# In[337]:


y_predict_LR = classifier_SVC.predict(titanic_test_n)


# In[338]:


pd.DataFrame(y_predict_LR).to_csv("survival_pred.csv")

