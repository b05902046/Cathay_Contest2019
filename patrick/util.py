import pandas as pd
from sklearn.model_selection import train_test_split

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

def balancedTrainVali(feature, label, test_size=0.1, random_state=33, balance=True):
    x_train, x_vali, y_train, y_vali = train_test_split(feature, label, test_size=test_size, random_state=random_state)
    if not balance: return x_train, x_vali, y_train, y_vali

    xy_train = pd.concat([x_train, y_train], axis=1, join='outer')

    train_N, train_Y = xy_train[xy_train['Y1']==0], xy_train[xy_train['Y1']==1]
    N_num, Y_num = len(train_N), len(train_Y)
    if N_num >= Y_num:
        train_Y = pd.concat([train_Y] * int(round(N_num/Y_num)), ignore_index=True)
    else:
        train_N = pd.concat([train_N] * int(round(Y_num/N_num)), ignore_index=True)
    xy_train = pd.concat([train_N, train_Y], ignore_index=True)
    x_train, y_train = xy_train.drop(['Y1'], axis=1), xy_train['Y1']
    return x_train, x_vali, y_train, y_vali

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