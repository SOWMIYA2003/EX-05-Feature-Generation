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
# DataSet 1 - data.csv
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
### Initial DataFrame (data.csv):
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

# DataSet 2 - Encoding Data.csv
## CODE
```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
oe=OrdinalEncoder()
oe.fit_transform(df[["ord_2"]])
temp=['Cold','Warm','Hot']
enc=OrdinalEncoder(categories=[temp])
enc
enc.fit_transform(df[['ord_2']])
df1=df.copy()
df1["ord_2"]=enc.fit_transform(df[["ord_2"]])
df1
oe1=OrdinalEncoder()
oe.fit_transform(df[["nom_0"]])
color=['Green','Blue','Red']
enc1=OrdinalEncoder(categories=[color])
enc1
enc1.fit_transform(df[['nom_0']])
df2=df1.copy()
df2["nom_0"]=enc1.fit_transform(df[["nom_0"]])
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

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df5=pd.DataFrame(scaler.fit_transform(df4),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df5

from sklearn.preprocessing import StandardScaler
Stdscaler=StandardScaler()
df6=pd.DataFrame(Stdscaler.fit_transform(df4),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df6

from sklearn.preprocessing import MaxAbsScaler
maxabsscaler=MaxAbsScaler()
df7=pd.DataFrame(maxabsscaler.fit_transform(df4),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df7

from sklearn.preprocessing import RobustScaler
rscaler = RobustScaler()
df8=pd.DataFrame(rscaler.fit_transform(df4),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df8
```
## OUTPUT
### Initial DataFrame (Encoding Data.csv)
![op](./1a.png)
### Feature Generation Process:
```
1.Ordinal Encoder.
2.Binary Encoder.
3.One Hot Encoder.
```
### Applying Ordinal Encoding Method in column - ord_2:
![op](./1b.png)
![op](./1c.png)
### After applying Ordinal Encoding Method in column - ord_2:
![op](./1d.png)
### Applying Ordinal Encoding Method in column - nom_0:
![op](./1e.png)
![op](./1f.png)
### After applying Ordinal Encoding Method in column - nom_0:
![op](./1g.png)
### Applying Binary Encoding Method in column - bin_1:
![op](./1h.png)
### After applying Binary Encoding in column- bin_1:
![op](./1i.png)
### Applying Binary Encoding Method in column - bin_2:
![op](./1l.png)
### After applying Binary Encoding in column- bin_2:
![op](./1m.png)
### Final DataSet after applying Encoding Methods:
![op](./1aa.png)
### Feature Scaling  Techniques:
```
1.Min-Max Scaler.
2.Standard Scaler.
3.Max Abs Scaler.
4.Robust Scaler.
```
### Feature Scaling - Min-Max Scaler Technique:
![op](./1n.png)
### Feature Scaling - Standard Scaler Technique:
![op](./1o.png)
### Feature Scaling - Max Abs Scaler Technique:
![op](./1p.png)
### Feature Scaling - Robust Scaler Technique:
![op](./1q.png)

# DataSet 3 - titanic_dataset.csv
## CODE
```
import pandas as pd
df=pd.read_csv("titanic_dataset.csv")
df
df.drop("Name",axis=1,inplace=True)
df
df.drop("Cabin",axis=1,inplace=True)
df
df.drop("Ticket",axis=1,inplace=True)
df
df.info()
df.isnull().sum()
df["Age"]=df["Age"].fillna(df["Age"].median())
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])
df.boxplot()
df.isnull().sum()
df

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
oe=OrdinalEncoder()
oe.fit_transform(df[["Embarked"]])
embark=['S','C','Q']
enc=OrdinalEncoder(categories=[embark])
enc
enc.fit_transform(df[['Embarked']])
df1=df.copy()
df1["Embarked"]=enc.fit_transform(df[["Embarked"]])
df1
pip install category_encoders
from category_encoders import BinaryEncoder
be=BinaryEncoder()
newdata=be.fit_transform(df1["Sex"])
newdata
df2=df1.copy()
df2["Sex"]=be.fit_transform(df1[["Sex"]])
df2

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df3=pd.DataFrame(scaler.fit_transform(df2),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df3

from sklearn.preprocessing import StandardScaler
Stdscaler=StandardScaler()
df4=pd.DataFrame(Stdscaler.fit_transform(df2),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df4

from sklearn.preprocessing import MaxAbsScaler
maxabsscaler=MaxAbsScaler()
df5=pd.DataFrame(maxabsscaler.fit_transform(df2),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df5

from sklearn.preprocessing import RobustScaler
rscaler = RobustScaler()
df6=pd.DataFrame(rscaler.fit_transform(df2),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df6
```
## OUTPUT
### Initial DataFrame - (titanic_dataset.csv):
![op](./2a.png)
### Droping "Name" column from DataFrame:
![op](./2b.png)
### Droping "Cabin" and "Ticket" column from DataFrame:
![op](./2c.png)
### Non Null-data Count:
![op](./2d.png)
### Sum of null data present in each column:
![op](./2e.png)
### Handling Null data in Column "Age" and "Embarked" :
![op](./0s.png)
### Data Frame after removing column - Age,Cabin,Ticket:
![op](./2h.png)
### Feature Generation Process:
```
1.Ordinal Encoder.
2.Binary Encoder.
3.One Hot Encoder.
```
### Applying Ordinal Encoding Method in column - Embarked:
![op](./2i.png)
![op](./0sh.png)
### After applying Ordinal Encoding Method in column - Embarked:
![op](./0am.png)
### Applying Binary Encoding Method in column - Sex:
![op](./2j.png)
### After applying Binary Encoding in column- Sex:
![op](./2k.png)
### Feature Scaling  Techniques:
```
1.Min-Max Scaler.
2.Standard Scaler.
3.Max Abs Scaler.
4.Robust Scaler.
```
### Feature Scaling - Min-Max Scaler Technique:
![op](./2s.png)
### Feature Scaling - Standard Scaler Technique:
![op](./2ha.png)
### Feature Scaling - Max Abs Scaler Technique:
![op](./2m.png)
### Feature Scaling - Robust Scaler Technique:
![op](./2ra.png)
## RESULT:
Feature Generation process and Feature Scaling process is applied to the given data frame sucessfully.