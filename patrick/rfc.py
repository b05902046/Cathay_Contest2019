##Use no NaN data to train, and split the data to 8:2
##Model used: Random Forest Classifier
import util

train_feature, train_label = util.getData('../newtrain.csv', False)

#x_train, x_vali, y_train, y_vali = balancedTrainVali(train_feature, train_label, test_size=0.1, random_state=33, balance=True)
x_train, x_vali, y_train, y_vali = util.balancedTrainVali(train_feature, train_label, test_size=0.1, random_state=33, balance=False)
print(len(x_train), len(x_train.columns))
print(len(x_vali), len(x_vali.columns))

#from sklearn.ensemble import RandomForestClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
#from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
import numpy as np

#rfc = RandomForestClassifier()
rfc = ComplementNB()
rfc.fit(x_train, y_train)

print('train')
ypred_train = rfc.predict(x_train)
util.showAcc(ypred_train, y_train)

print('\nvalidation')
ypred_vali = rfc.predict(x_vali)
util.showAcc(ypred_vali, y_vali)


test_feature, test_cusid = util.getData('../newtest.csv', True)
predicted = rfc.predict(test_feature)

util.writeResult('../output.csv', test_cusid, predicted)
