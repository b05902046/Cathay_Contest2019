##Use no NaN data to train, and split the data to 8:2
##Model used: Random Forest Classifier
import util
from imblearn.over_sampling import SMOTE, ADASYN

train_feature, train_label = util.getData('../newtrain.csv', False)

#x_train, x_vali, y_train, y_vali = balancedTrainVali(train_feature, train_label, test_size=0.1, random_state=33, balance=True)
x_tr, x_vali, y_tr, y_vali = util.balancedTrainVali(train_feature, train_label, test_size=0.1, random_state=33, balance=False)

#x_train, y_train = SMOTE().fit_resample(x_train, y_train)
x_train, y_train = ADASYN().fit_resample(x_tr, y_tr)
print(x_tr.shape, len(y_tr))

print(x_train.shape, len(y_train))
print(x_vali.shape, len(y_vali))

#from sklearn.ensemble import RandomForestClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
#from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.svm import LinearSVC

#rfc = RandomForestClassifier()
svc = LinearSVC()
svc.fit(x_train, y_train)

print('train')
ypred_train = svc.predict(x_train)
util.showAcc(ypred_train, y_train)

print('\nvalidation')
ypred_vali = svc.predict(x_vali)
util.showAcc(ypred_vali, y_vali)


test_feature, test_cusid = util.getData('../newtest.csv', True)
predicted = svc.predict(test_feature)

util.writeResult('../output.csv', test_cusid, predicted)
