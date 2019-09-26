import util
import pandas as pd
import numpy as np

train_feature, train_label = util.getData('../newtrain.csv', False)
print(train_feature.shape, train_label.shape)
print('ratio of 1:', np.sum(train_label)/len(train_label))

Xtr, Ytr = util.balanceData(train_feature, train_label, 'ADASYN')
print(Xtr.shape, Ytr.shape)
print('ratio of 1:', np.sum(Ytr)/len(Ytr))

balanced = np.concatenate((Xtr,np.array([Ytr]).T), axis=1)
pd.DataFrame(balanced).to_csv('balanced.csv', header=None, index=None)
