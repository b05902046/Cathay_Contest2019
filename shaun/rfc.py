##Use no NaN data to train, and split the data to 8:2
##Model used: Random Forest Classifier

import pandas as pd

#Load data
df = pd.read_csv("../newtrain.csv", encoding= "utf-8")

#one-hot encode and extract labels and features
label = df['Y1']
label = label.map({"N":0, "Y":1})
feature = df.drop(["CUS_ID", "Y1"], axis=1)
feature = pd.get_dummies(feature)

print(label)
print(feature)

#
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size = 0.2, random_state = 33)

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)

print(rfc.score(x_test, y_test))
