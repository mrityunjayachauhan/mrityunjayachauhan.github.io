#!/usr/bin/env python
# coding: utf-8

# ## Project Name: House Prices: Advanced Regression Techniques
# 
# The main aim of this project is to predict the house price based on various features 
# 
# Dataset to downloaded from the below link
#     [Advanced House Price](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)

# ## All the Lifecycle In A Data Science Projects
# 1. Data Analysis / Data Pre-Processing
# 2. Feature Engineering
# 3. Feature Selection
# 4. Model Building
# 5. Model Evaluation
# 6. Model Deployment

# ## 1. Data Analysis Phase
#  Main aim is to understand more about the data and deep dive into the basic data analytics processes.
#  #### In Data Analysis We will Analyze To Find out the below stuff
# 1. Checking for Missing Values
# 2. All The Numerical Variables
# 3. Distribution of the Numerical Variables
# 4. Categorical Variables
# 5. Cardinality of Categorical Variables
# 6. Outliers
# 7. Relationship between independent and dependent feature(SalePrice)
# 
# 

# In[1]:


# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from glob import glob
import os


# In[2]:


# Reading dataset
direc = os.chdir(r"C:\Advanced-House-Price-Prediction")
filename = [file for file in glob('*.{}'.format('csv'))]
df = pd.concat([pd.read_csv(file) for file in filename if 'train' in file],ignore_index=True)
#pd.options.display.max_columns=100
pd.pandas.set_option('display.max_columns',None)


# In[3]:


df.head()


# In[4]:


print("Shape: ",df.shape)


# ##### Missing Values

# In[5]:


# Printing the percentage of nan values present in each features
#1. selecting the features where nan value is present and storing in a list
features_nan=[features for features in df.columns if df[features].isnull().sum()>1]
print('Number of features having NaN:',len(features_nan))
#2. printing features name with % missing value
for features in features_nan:
    print(features,":", np.round(df[features].isnull().mean()*100,2), '% missing values')


# ##### Finding relationship between missing values and sales price

# In[6]:


for feature in features_nan:
    df_copy = df.copy()
    # creating variable to flag NaN observation with 1 or 0
    df_copy[feature]=np.where(df_copy[feature].isnull(),1,0)
    df_copy.groupby(feature)['SalePrice'].median().plot.bar(color=['green','orange'])
    plt.title(feature)
    plt.show()


# There is significant relationship between features having missing values with the target variable. Therefore we not delete the missing records.

# In[7]:


# Finding Numerical Variables
numerical = [features for features in df.columns if df[features].dtypes !='object']
print("Number of Numerical variables:", len(numerical))
df[numerical].head()


# #### Temporal Variables(Eg: Datetime Variables)
# 

# In[8]:


year_feature = [features for features in df.columns if 'Yr' in features or 'Year' in features]
df[year_feature].head()


# In[9]:


# Analyse SalePrice with Year of Sold
df.groupby('YrSold')['SalePrice'].median().plot(color='g')
plt.title('House Price Vs YearSold')
plt.xlabel('Year_Sold')
plt.ylabel('Sale Price Median')


# - As age of the house increases the selling price of the comodities falls down because of deprecition.

# In[10]:


# Comapring all years with sale price
for feature in year_feature:
    if feature != 'YrSold':
        df_copy = df.copy()
        df_copy[feature]=df_copy['YrSold']-df[feature]
        plt.scatter(df_copy[feature],df['SalePrice'],color='green')
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.title('Sale Price Vs {}'.format(feature))
        plt.show()


# - Sales price will decrease as the age of house/garage increases because of depreciation. 

# In[11]:


# Numerical variable are of 2 types : Continuous & Discrete
discrete = [features for features in numerical if len(df[features].unique())<25 and features not in year_feature+['Id'] ]
print("# Discrete variables:",len(discrete))


# In[12]:


df[discrete].head()


# In[13]:


# finding relationship between discrete data and saleprice
from numpy import array
for feature in discrete:
    df_copy =df.copy()
    dis=df_copy.groupby(feature)['SalePrice'].median().reset_index()
    sns.barplot(x=dis[feature],y=dis['SalePrice'])
    plt.xlabel(feature)
    plt.ylabel('SalePrice median')
    plt.show()
    


# - In this dataset, we have 17 discete variable having a number of categories. From above graph it is clear that the sales price depends on the categories. Few of the discrete variables (like OverallQual, FullBath) has direct relationship with the sales price.

# In[14]:


# continuous features
continuous = [features for features in numerical if features not in discrete+year_feature+['Id']]
print("# continuous features", len(continuous))


# In[15]:


for feature in continuous:
    df_copy=df.copy()
    sns.histplot(df_copy[feature],bins=25,kde=True,color='purple',line_kws={'linestyle':'--'})
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.show()


# - Above histogram is the evidence that the data of all the continuous features are skewed and not follow the gaussian distribution.

# In[16]:


# Using Logarithmic transformation
for feature in continuous:
    df_copy=df.copy()
    if 0 in df_copy[feature].unique():
        pass
    else:
        df_copy[feature]=np.log(df_copy[feature])
        df_copy['SalePrice']=np.log(df_copy['SalePrice'])
        plt.scatter(df_copy[feature],df_copy['SalePrice'],color='g')
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.title(feature)
        plt.show()


# #### Outliers

# In[17]:



for feature in continuous:
    df_copy=df.copy()
    if 0 in df_copy[feature].unique():
        pass
    else:
        df_copy[feature]=np.log(df_copy[feature])
        plt.boxplot(df_copy[feature],whis=True)
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()


# - Since the data are skewed and not normally distributed it is obvious that the continuos feature would contains outliers. Outliers are those values which is completely different from the remaining set of data.

# #### Categorical Variables

# In[18]:


categorical = [features for features in df.columns if df[features].dtypes=='object']
print("# of categorical features:", len(categorical))


# In[19]:


df[categorical].head()


# In[20]:


for features in categorical:
    print("Feature {} has {} number of categories.".format(features,len(df[features].unique())))


# In[21]:


# relationship between categorical features and dependent variable
for features in categorical:
    df_copy=df.copy()
    df_copy.groupby(features)['SalePrice'].median().plot.bar(color=['orange','green','blue','red','violet','brown','cyan','yellow'])
    plt.xlabel(features)
    plt.ylabel('SalePrice median')
    plt.show()


# ### Feature Engineering

# We will be performing all the below steps in Feature Engineering
# 
# 1. Dealing with Missing values
# 2. Dealing with Temporal variables
# 3. Categorical variables: Remove of rare labels
# 4. Standarise the values of the variables to the same range

# #### Missing Values

# In[22]:


# Categorical variables
categorical_nan =[features for features in df.columns if df[features].isnull().sum()>1 and df[features].dtypes=='object']
for features in categorical_nan:
    print(features,":",np.round(df[features].isnull().mean()*100,2),"missing values")


# In[23]:


# Replacing missing value with "Missing" in categorical variable
def replace_nan_categorical(df,categorical_nan):
    data=df.copy()
    data[categorical_nan]=data[categorical_nan].fillna('Missing')
    return data

df = replace_nan_categorical(df,categorical_nan)
df[categorical_nan].isnull().sum()


# In[24]:


df.head()


# In[25]:


# Numerical Variables
numerical_nan=[features for features in df.columns if df[features].isnull().sum()>1 and df[features].dtypes!='O']

for features in numerical_nan:
    print(features,":",np.round(df[features].isnull().mean()*100,2),'missing values')


# In[26]:


# Replacing missing values in numerical variables
for features in numerical_nan:
    #replace the missing value according--
    # for normal distribution data use mean and for skewed data or data containing outliers use median
    median_value = df[features].median()
    # creating new feature to flag missing values
    df[features+'nan']=np.where(df[features].isnull(),1,0)
    df[features].fillna(median_value,inplace=True)
    
df[numerical_nan].isnull().sum()    


# In[27]:


df.head()


# In[28]:


# Temporal variable
temporal = ['YearBuilt','YearRemodAdd','GarageYrBlt']
for features in temporal:
    df[features]=df['YrSold']-df[features]
    
df[temporal].head()    


# ### Numerical Variables

#  Since numerical variables are skewed we will perform log normal distribution

# In[29]:


num_features=['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']

for feature in num_features:
    df[feature]=np.log(df[feature])
    
df.head()


# ## Handling Rare Categorical Feature
# 
# We will remove categorical variables that are present less than 1% of the observations

# In[30]:


categorical_features=[feature for feature in df.columns if df[feature].dtypes=='O']

for feature in categorical_features:
    temp=df.groupby(feature)['SalePrice'].count()/len(df)
    temp_df=temp[temp>0.01].index
    df[feature]=np.where(df[feature].isin(temp_df),df[feature],'Rare_var')
    
df.head()
    


# In[31]:


# Label Encoding for categorical variable
for feature in categorical_features:
    labels_ordered=df.groupby([feature])['SalePrice'].mean().sort_values().index
    labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}
    df[feature]=df[feature].map(labels_ordered)


# In[32]:


df.head()


# ### Features Scaling

# In[33]:


scaling_features =[features for features in df.columns if features not in['Id','SalePrice']]
len(scaling_features)


# *** Always remember to split the dataset into train and test to avoid data leakage before feature engineering

# In[34]:


independent= [features for features in df.columns if features not in ['Id','LotFrontagenan','MasVnrAreanan','GarageYrBltnan','SalePrice']]
X=df[independent]
Y=df['SalePrice']


# In[35]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,Y,random_state=42,test_size=0.2)


# In[36]:


# Normalizating the data since the data is not normally distributed and contains outliers
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()


# In[37]:


# Normalize only independent features
X_Train=scaler.fit_transform(X_train)
X_Test=scaler.fit_transform(X_test)


# In[38]:


x_train=pd.DataFrame(X_Train,columns=independent)
x_train.head()


# In[39]:


x_test=pd.DataFrame(X_Test,columns=independent)
x_test.head()


# ### Apply Feature Selection
# First, I have specified the Lasso Regression model, and I have selected a suitable alpha (equivalent of penalty).
# The bigger the alpha the less features that will be selected.
# 
# Then I use the selectFromModel object from sklearn, which will select the features were coefficients are non-zero.
# 
# 

# In[42]:


## Feature slection
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
feature_sel = SelectFromModel(Lasso(alpha=0.001, random_state=0)) # remember to set the seed, the random state in this function
feature_sel.fit(x_train, y_train)


# In[43]:


# let's print the number of total selected features

# this is how we can make a list of the selected features
selected_feature = x_train.columns[(feature_sel.get_support())]

# let's print some stats
print('total features: {}'.format((x_train.shape[1])))
print('selected features: {}'.format(len(selected_feature)))
print('features with coefficients shrank to zero: {}'.format(np.sum(feature_sel.estimator_.coef_ == 0)))


# ### Model Building

# In[44]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
lasso=Lasso()


# In[45]:


alpha= np.logspace(-4,4,9)
param={'alpha':alpha}
search=GridSearchCV(lasso,param,cv=5)
search.fit(x_train,y_train)


# In[46]:


print("Best alpha:",search.best_params_['alpha'])
print("Best Score:",search.best_score_)


# In[47]:


x_train_final=x_train[selected_feature]
x_test_final=x_test[selected_feature]


# In[48]:


from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train_final,y_train)


# In[49]:


y_pred=reg.predict(x_test_final)


# In[50]:


# R-squred
from sklearn.metrics import r2_score,mean_squared_error
print("R-Squared:",np.round(r2_score(y_test,y_pred),4))


# In[51]:


np.round(reg.score(x_test_final,y_test),4)


# In[52]:


# Testing error
print("MSE:",mean_squared_error(y_test,y_pred))
print("RMSE:",np.sqrt(mean_squared_error(y_test,y_pred)))


# In[53]:


# Training error
print("MSE:",mean_squared_error(y_train,reg.predict(x_train_final)))


# - The model accuracy is approx 90% with low bias and low variance. Hence, we can say that the model is now generalized.

# In[ ]:




