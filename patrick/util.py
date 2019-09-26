import pandas as pd

def getData(path, isTest=False):
    df = pd.read_csv(path, encoding='utf-8').dropna(axis=1)
    #one-hot encode and extract labels and features
    if isTest:
        feature = df.drop(['CUS_ID'], axis=1)
        return pd.get_dummies(feature), df['CUS_ID']
    else:
        feature = df.drop(['CUS_ID', 'Y1'], axis=1)
        feature = pd.get_dummies(feature)
        label = df['Y1'].map({'N':0, 'Y':1})
        return feature, label

def writeResult(path, customIDs, predictions):
    df = pd.DataFrame(predictions, columns=['Ypred'])
    pd.concat([customIDs, df], axis=1, join='outer').to_csv(path, index=False)

from imblearn.over_sampling import SMOTE, ADASYN

def balanceData(feature, label, balance=None):
    if balance==None:
        return feature, label
    elif balance == 'SMOTE':
        return SMOTE().fit_resample(feature, label)
    elif balance == 'ADASYN':
        return ADASYN().fit_resample(feature, label)
    else:
        raise ValueError

from sklearn.metrics import confusion_matrix
import numpy as np

def showAcc(predicted, true):
    cm = confusion_matrix(list(true), predicted)
    Nnum, Ynum = np.sum(cm, axis=1)
    totalAcc = float(cm[0][0]+cm[1][1])/(Nnum+Ynum)
    accCM = np.array(cm, dtype=np.float32)
    accCM[0], accCM[1] = accCM[0]/Nnum, accCM[1]/Ynum
    print('totalAcc:', totalAcc)
    np.set_printoptions(formatter={'float':'{:0.3f}'.format})
    print(accCM)
