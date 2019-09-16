##Use no NaN data to train, and split the data to 8:2
##Model used: Random Forest Classifier
import pandas as pd

def getData(path, isTest=False):
    df = pd.read_csv(path, encoding='utf-8').dropna(axis=1)
    #one-hot encode and extract labels and features
    if isTest:
        feature = df.drop(['CUS_ID'], axis=1)
        return pd.get_dummies(feature)
    else:
        feature = df.drop(['CUS_ID', 'Y1'], axis=1)
        feature = pd.get_dummies(feature)
        label = df['Y1'].map({'N':0, 'Y':1})
        return feature, label

train_feature, train_label = getData('../newtrain.csv', False)

from sklearn.model_selection import train_test_split

x_train, x_vali, y_train, y_vali = train_test_split(train_feature, train_label, test_size = 0.1, random_state = 33)
print(len(x_train))
print(len(x_vali))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import numpy as np

rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)

def showAcc(predicted, true):
    cm = confusion_matrix(list(true), predicted)
    Nnum, Ynum = np.sum(cm, axis=1)
    totalAcc = float(cm[0][0]+cm[1][1])/(Nnum+Ynum)
    accCM = np.array(cm, dtype=np.float32)
    accCM[0], accCM[1] = accCM[0]/Nnum, accCM[1]/Ynum
    print('totalAcc:', totalAcc)
    np.set_printoptions(formatter={'float':'{:0.3f}'.format})
    print(accCM)

print('train')
ypred_train = rfc.predict(x_train)
showAcc(ypred_train, y_train)

print('\nvalidation')
ypred_vali = rfc.predict(x_vali)
showAcc(ypred_vali, y_vali)
