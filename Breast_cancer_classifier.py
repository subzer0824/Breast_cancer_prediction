#!/usr/bin/env python
# coding: utf-8

# ## INTRODUCTION
# (Dataset Source - Kaggle)

# #### About the Dataset
# The Breast Cancer (Wisconsin) Diagnosis dataset contains a set of 30 features which describe the characteristics of cell nuclei that are computed from digitized images of a fine needle aspirate (FNA) of a breast mass. 

# We will analyze the features to understand the predictive values for diagnosis
# 

# We will carry out the following three steps:

# A. Data Analysis

# B. Feature Engineering

# C. Model Building and Testing

# ## A. Data Analysis

# In[ ]:


#Necessary Imports for exploratory data analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Read the data
data_original = pd.read_csv('data.csv')


# In[ ]:


#Get an idea of the data
data_original.head()


# In[4]:


#Shape of the data
data_original.shape


# In[5]:


dataset = data_original.copy()  #create a copy to preserve original data


# In[6]:


dataset.columns


# We will include the following:
# 1. Missing Values 
# 2. Categorical features
# 3. Numerical features
# 4. Outliers
# 5. Relationship between independent and dependent features

# ### 1. Missing Values

# In[7]:


na_features = [feature for feature in dataset.columns if dataset[feature].isnull().sum()>0]

for feature in na_features :
    print(feature,': ',np.round(dataset[feature].isnull().mean(),4), '% missing values')


# In[8]:


#Unnamed: 32 feature contains an empty column so we drop it
#id is also of not any use to us so we also drop it
dataset = dataset.drop('Unnamed: 32',axis = 1).drop('id',axis = 1)


# ### 2. Categorical Features

# In[9]:


categorical_features = [feature for feature in dataset.columns if dataset[feature].dtypes == 'O']


# In[10]:


len(categorical_features)


# In[11]:


dataset[categorical_features].head()


# In[182]:


print('Malignant cases: ',((dataset['diagnosis'] == 'M')*1).sum())
print('Benign cases: ',((dataset['diagnosis'] == 'B')*1).sum())


# #### The dataset is balanced, so accuracy is a dependable metric

# ### 3. Numerical Features

# In[12]:


num_features = [feature for feature in dataset.columns if dataset[feature].dtypes != 'O']


# In[13]:


len(num_features)


# In[14]:


dataset[num_features].head()


# ### Numerical features are of two types:

# ### a. Discrete Feature

# In[15]:


discrete_features = [feature for feature in num_features if len(dataset[feature].unique())<25 ]
discrete_features


# There are no discrete features

# ### b. Continuous Features

# In[16]:


continuous_features  = [feature for feature in num_features if feature not in discrete_features]
continuous_features


# In[17]:


len(continuous_features)


# All features are continuous

# In[18]:


dataset[continuous_features].head()


# In[19]:


sns.pairplot(dataset.iloc[1:],hue='diagnosis')


# #### A common trend observed is that higher mean value of the tumour's dimension is found in malignant tumour

# In[20]:


# Correlatioon Heatmap
corrmat = dataset[continuous_features].corr(method='pearson')
f, ax = plt.subplots(figsize =(10, 9))
sns.heatmap(corrmat,cmap='YlGnBu',linewidths=0.1)


# ### 4. Outliers

# In[21]:


# Outliers in Continuous Features
for feature in continuous_features:    
    data=dataset.copy()
    if 0 in data[feature].unique():  #log cannot take 0
        pass
    else:
        data[feature] = np.log(data[feature])
        data.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()


# ## B. Feature Engineering

# ### 1.Feature Scaling

# In[22]:


dataset.head()


# In[23]:


feature_scale = [feature for feature in dataset.columns if feature not in ['diagnosis']]


# In[24]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(dataset[feature_scale])


# In[25]:


scaled_data = scaler.transform(dataset[feature_scale])


# In[26]:


data = pd.concat([data_original[['id','diagnosis']].reset_index(drop = True),pd.DataFrame(scaled_data,
                  columns = feature_scale)],axis = 1)


# In[28]:


data.to_csv("Scaled_data.csv",index=False)


# ### 2.Feature Selection

# In[29]:


dataset = pd.read_csv("Scaled_data.csv")


# In[30]:


dataset.head()


# In[31]:


y = dataset[['diagnosis']]   #Dependent Feature
X = dataset.drop(['id','diagnosis'],axis=1)   #Independent Feature


# In[32]:


y[y=='M'] = 1
y[y=='B'] = 0


# In[33]:


from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel


# In[34]:


feature_sel_model = SelectFromModel(Lasso(alpha=0.005,random_state=0))
feature_sel_model.fit(X,y)


# In[35]:


feature_sel_model.get_support()


# In[36]:


selected_features = X.columns[feature_sel_model.get_support()]


# In[37]:


print('Selected Features: ',len(selected_features))
selected_features


# In[38]:


X = X[selected_features]


# In[39]:


X.head()


# # C. Model Fitting And Testing

# In[40]:


#imports for model fitting and testing
import sys
import scipy
import sklearn
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
from sklearn import preprocessing


# In[111]:


# Splitting dataset into Training Set and Testing Set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)


# In[112]:


X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()


# ## 1. ANN(Artificial Neural Network)

# In[44]:


import keras
from keras.models import Sequential
from keras.layers import Input,Dense, Dropout, Activation, BatchNormalization
from keras.optimizers import SGD


# In[160]:


model = Sequential()

#Input Layer
model.add(Dense(64, input_dim = 5)) 
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Dropout(0.2))

#First Hidden Layer
model.add(Dense(32))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Dropout(0.2))

#Second Hidden layer
model.add(Dense(32))
model.add(Activation('softmax'))

#Output Layer
model.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))


#Compile
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[161]:


model.fit(X_train,y_train,epochs=20, batch_size=32, validation_data=(X_test,y_test))


# In[164]:


# Predicting the test set results

y_pred = model.predict(X_test)
y_pred = (y_pred >= 0.5)*1

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test.astype(int),y_pred.astype(int))

sns.heatmap(conf_mat, annot = True)
print('accuracy: ',(accuracy_score(y_test.astype(int),y_pred.astype(int))*100).round(4),'%')


# ## 2. KNN(K nearest neighbour)

# In[148]:


def acc_check(name, model):
    kfold = model_selection.KFold(n_splits= 10, random_state= 0)
    cv_results = model_selection.cross_val_score(model,X_train,y_train.astype(int),cv= kfold,scoring = 'accuracy')
    print(name, ': cv_results:',cv_results.mean().round(4),' std: (' ,cv_results.std().round(4),')')
    
#KNN
acc_check("KNN",KNeighborsClassifier(n_neighbors=5))
acc_check("SVM",SVC())


# In[155]:


#Predictions on validation set

def pred_val(name,model):
    model.fit(X_train,y_train.astype(int))
    predictions = model.predict(X_test)
    print(name,'- accuracy_score: ',accuracy_score(y_test.astype(int),predictions).round(4)*100,'%')
    print(classification_report(y_test.astype(int),predictions))
    
    


# In[188]:


#KNN prediction
pred_val("KNN",KNeighborsClassifier(n_neighbors=5))


# ##  3. SVM(Support Vector Machine)

# In[198]:


pred_val("SVM",SVC())


# ## 4. Random Forest Classifier

# In[197]:


#Random Forest Classifier prediction
from sklearn.ensemble import RandomForestClassifier

pred_val("RFC",RandomForestClassifier(random_state=0))


# ## Conclusion

# 1. In order to predict the class of the tumour we need only 5   features viz. concave points_mean, radius_worst, texture_worst, smoothness_worst, concave points_worst.

# 2. Most of the outliers fall in the class of Malignant tumour

# 3. Accuracy scores of the models are as follows:
ANN : 97.3684 %
KNN : 97.37 %
SVM : 94.74 %
RFC : 96.49 %