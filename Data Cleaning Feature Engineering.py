import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv(r'../input/house-prices-advanced-regression-techniques/test.csv')
y_all = np.log1p(train['SalePrice'])
train = train.drop('SalePrice', axis=1)
train = pd.concat([train,test], axis=0).reset_index(drop=True)
drop_columns_due_to_missing = [col for col in train.columns if train[col].isnull().sum()/len(train[col])>0.1]

train = train.drop(drop_columns_due_to_missing+['Id'],axis=1)
quantity = [atr for atr in train.columns if train[atr].dtypes != 'object']
quality = [atr for atr in train.columns if train[atr].dtypes == 'object']
train[quality] = train[quality].fillna('Missing')

for atr in quantity:
    train[atr] = train[atr].fillna(train[atr].mean())
quality_dummies_col = [col for col in quality if train[col].value_counts().count() <= 6]
quality_labelencoder_col = [col for col in quality if train[col].value_counts().count() > 6]

for col in quality_dummies_col:
    train = pd.concat([train, pd.get_dummies(train[col],prefix=col)], axis=1)
    train = train.drop(col, axis=1)

le = preprocessing.LabelEncoder()
for col in quality_labelencoder_col:
    train[col] = le.fit_transform(train[col])

train[quantity] = preprocessing.StandardScaler().fit_transform(train[quantity])
x1, x2, y1, y2 = train_test_split(train[train.index<=1459], y_all, test_size=0.4)