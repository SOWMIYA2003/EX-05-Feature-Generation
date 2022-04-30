# EX-05-Feature-Generation


## AIM
To read the given data and perform Feature Generation process and save the data to a file. 

## Explanation
Feature Generation (also known as feature construction, feature extraction or feature engineering) is the process of transforming features into new features that better relate to the target.
 

## ALGORITHM
### STEP 1
Read the given Data.
### STEP 2
Clean the Data Set using Data Cleaning Process.
### STEP 3
Apply Feature Generation techniques to all the feature of the data set.
### STEP 4
Save the data to the file.



## Feature Generation Process and Feature Scaling Techniques:

### Ordinal Encoding

In ordinal encoding, each unique category value is assigned an integer value. For example, “red” is 1, “green” is 2, and “blue” is 3. This is called an ordinal encoding or an integer encoding and is easily reversible. Often, integer values starting at zero are used.

### Binary Encoding

Initially categories are encoded as Integer and then converted into binary code, then the digits from that binary string are placed into separate columns. for eg: for 7 : 1 1 1. This method is quite preferable when there are more number of categories.

### One Hot Encoding

One Hot Encoding is a common way of preprocessing categorical features for machine learning models. This type of encoding creates a new binary feature for each possible category and assigns a value of 1 to the feature of each sample that corresponds to its original category.

### Min-Max Scaler

In machine learning, MinMaxscaler is a scaling algorithms for continuous variables. The MinMaxscaler is a type of scaler that scales the minimum and maximum values to be 0 and 1 respectively.
MinMax Scaler shrinks the data within the given range, usually of 0 to 1. It transforms data by scaling features to a given range. It scales the values to a specific value range without changing the shape of the original distribution.

### Standard Scaler

In machine learning, StandardScaler is a scaling algorithms for continuous variables.StandardScaler removes the mean and scales each feature/variable to unit variance. This operation is performed feature-wise in an independent way. StandardScaler can be influenced by outliers (if they exist in the dataset) since it involves the estimation of the empirical mean and standard deviation of each feature.

### Max Abs Scaler

Maximum absolute scaling scales the data to its maximum value; that is, it divides every observation by the maximum value of the variable: The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.

### Robust Scaler

One approach to standardizing input variables in the presence of outliers is to ignore the outliers from the calculation of the mean and standard deviation, then use the calculated values to scale the variable. This is called robust standardization or robust data scaling.
Scale features using statistics that are robust to outliers. 
This Scaler removes the median and scales the data according to the quantile range (defaults to IQR: Interquartile Range). The IQR is the range between the 1st quartile (25th quantile) and the 3rd quartile (75th quantile).

## CODE
```
import pandas as pd
df=pd.read_csv("data.csv")
df
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
oe=OrdinalEncoder()
oe.fit_transform(df[["Ord_1"]])
temp=['Cold','Warm','Hot','Very Hot']
enc=OrdinalEncoder(categories=[temp])
enc
enc.fit_transform(df[['Ord_1']])
df1=df.copy()
df1["Ord_1"]=enc.fit_transform(df[["Ord_1"]])
df1
oe1=OrdinalEncoder()
oe.fit_transform(df[["Ord_2"]])
studies=['High School','Diploma','Bachelors','Masters','PhD']
enc1=OrdinalEncoder(categories=[studies])
enc1
enc1.fit_transform(df[['Ord_2']])
df2=df1.copy()
df2["Ord_2"]=enc1.fit_transform(df[["Ord_2"]])
df2
pip install category_encoders
from category_encoders import BinaryEncoder
be=BinaryEncoder()
newdata=be.fit_transform(df2["bin_1"])
newdata
df3=df2.copy()
df3["bin_1"]=be.fit_transform(df[["bin_1"]])
df3
be1=BinaryEncoder()
newdata2=be1.fit_transform(df3['bin_2'])
newdata2
df4=df3.copy()
df4["bin_2"]=be1.fit_transform(df[["bin_2"]])
df4
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
ohe.fit_transform(df4[['City']])
num=ohe.fit_transform(df4[['City']])
num
df4
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
le=LabelEncoder()
le.fit_transform(df4[['City']])
df5=df4.copy()
df5["City"]=le.fit_transform(df4[['City']])
df5

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df6=pd.DataFrame(scaler.fit_transform(df5),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df6

from sklearn.preprocessing import StandardScaler
Stdscaler=StandardScaler()
df7=pd.DataFrame(Stdscaler.fit_transform(df5),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df7

from sklearn.preprocessing import MaxAbsScaler
maxabsscaler=MaxAbsScaler()
df8=pd.DataFrame(maxabsscaler.fit_transform(df5),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df8

from sklearn.preprocessing import RobustScaler
rscaler = RobustScaler()
df9=pd.DataFrame(rscaler.fit_transform(df5),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df9

```
## OUPUT
### DataFrame (data.csv):
![op](./5a.png)

### Feature Generation Process:

Categorical-Column Encoding Methods : Methods to convert categorical data to numeric data.
```
1.Ordinal Encoder.
2.Binary Encoder.
3.One Hot Encoder.
```
### Applying Ordinal Encoding Method in column- Ord_1:
![op](./5b.png)
![op](./5c.png)
### After applying Ordinal Encoding Method in column- Ord_1:
![op](./5d.png)
### Applying Ordinal Encoding Method in column- Ord_2:
![op](./5e.png)
![op](./5f.png)
### After applying Ordinal Encoding Method in column- Ord_2:
![op](./5g.png)
### Applying Binary Encoding Method in column- bin_1:
![op](./5h.png)
### After applying Binary Encoding in column- bin_1:
![op](./5i.png)
### Applying Binary Encoding Method in column- bin_2:
![op](./5j.png)
### After applying Binary Encoding Method in column- bin_2:
![op](./5k.png)
### One Hot Encoding Method:
![op](./5l.png)
![op](./5mm.png)
### Final DataSet after applying Encoding Methods:
![op](./5m.png)

### Feature Scaling  Techniques:
Feature scaling is a method used to normalize the range of independent variables or features of data. In data processing, it is also known as data normalization and is generally performed during the data preprocessing step.
```
1.Min-Max Scaler.
2.Standard Scaler.
3.Max Abs Scaler.
4.Robust Scaler.
```

### Feature Scaling - Min-Max Scaler Technique:
![op](./5n.png)
### Feature Scaling - Standard Scaler Technique:
![op](./5o.png)
### Feature Scaling - Max Abs Scaler Technique:
![op](./5p.png)
### Feature Scaling - Robust Scaler Technique:
![op](./5q.png)

## RESULT:
Feature Generation process and Feature Scaling process is applied to the given data frame sucessfully.