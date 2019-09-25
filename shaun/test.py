import sys
print(sys.path)


import xgboost as xgb

tmp = xgb.XGBClassifier(max_dept=3, learning_rate=0.1, n_estimators=100)
