# Fraud-Detection

import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns

df=pd.read_csv('/content/Fraud.csv')

df.head()

#handling missing value
df.isnull().sum()

df_filled=df.fillna(df['oldbalanceDest'].mean())
df_filled=df.fillna(df['newbalanceDest'].mean())
df_filled=df.fillna(df['isFraud'].mean())
df_filled=df.fillna(df['isFlaggedFraud'].mean())
print(df_filled)

#Z score method to identify outliers
numeric_columns = df_filled.select_dtypes(include=[np.number]).columns
z_scores=np.abs(stats.zscore(df_filled[numeric_columns]))
outliers=np.where(z_scores>3)
print("Outliers detected at positions:",outliers)

#Removing outliers
# Create a boolean mask for rows without outliers
mask = np.ones(df_filled.shape[0], dtype=bool)
mask[outliers[0]] = False

# Filter out the outliers
df_no_outliers = df_filled[mask]
print("DataFrame without outliers:\n", df_no_outliers)


#Detecting Multicollinearity

df_no_outliers.dtypes.unique()
num=['int64','float64']
num_vars=list(df.select_dtypes(include=num))
num_vars
df_no_outliers=df_no_outliers[num_vars]
df_no_outliers.shape


df_no_outliers.isnull().sum()
df_no_outliers.dropna(inplace=True)
x=df_no_outliers.iloc[:,1:9]
y=df_no_outliers.iloc[:,-1]
x.head()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=123)
corrmatrix=x_train.corr()
corrmatrix
sns.heatmap(corrmatrix,annot=True)
def correlation(df,threshold):
  correlated_cols=set()
  corr_matrix=df.corr()
  for i in range (len(corr_matrix.columns)):
    for j in range (i):
      if abs(corr_matrix.iloc[i,j])>threshold:
        colname=corr_matrix.columns[i]
        correlated_cols.add(colname)
        return correlated_cols



correlation(x_train,0.6)
corr_feature=correlation(x_train,0.6)
corr_feature
x_train.drop(labels=corr_feature,axis=1,inplace=True)
x_test.drop(labels=corr_feature,axis=1,inplace=True)

#Exploring transaction type
df_filled.type.value_counts()

type = df_filled["type"].value_counts()
transactions = type.index
quantity = type.values

import plotly.express as px
figure = px.pie(df_filled,
             values=quantity,
             names=transactions,hole = 0.5,
             title="Distribution of Transaction Type")
figure.show()

# Checking correlation
numeric_columns = df_no_outliers.select_dtypes(include=[np.number])
correlation = numeric_columns.corr()
print(correlation["isFraud"].sort_values(ascending=False))

df_filled["type"] = df_filled["type"].map({"CASH_OUT": 1, "PAYMENT": 2,
                                 "CASH_IN": 3, "TRANSFER": 4,
                                 "DEBIT": 5})
df_filled["isFraud"] = df_filled["isFraud"].map({0: "No Fraud", 1: "Fraud"})
df_filled.head()

# splitting the data
from sklearn.model_selection import train_test_split
x = np.array(df_filled[["type", "amount", "oldbalanceOrg", "newbalanceOrig","newbalanceDest","oldbalanceDest","step"]])
y = np.array(df_filled[["isFraud"]])

# training a machine learning model
from sklearn.tree import DecisionTreeClassifier
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20, random_state=42)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
model.score(xtest, ytest)

# prediction
#features = [type, amount, oldbalanceOrg, newbalanceOrig, newbalanceDest, oldbalanceDest, step]
features = np.array([[3, 0, 1000, 10,5,0.2,0]])
print(model.predict(features))
