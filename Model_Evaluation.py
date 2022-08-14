import xgboost as xgb
import numpy as np
import os
import pandas as pd
import sklearn as sk
from sklearn import metrics
from definitions import CSV_PATH, XGB_MODEL_PATH

# dana = CSV_generate.data
# print(np.size(dana))
classes = ['sitting', 'standing', 'lying']
df = pd.read_csv(CSV_PATH + "/data.csv")

X = df.drop(' POSE', axis=1).copy().values  # spacje popraw potem
y = df[' POSE'].copy().values

X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size=0.2)

model = xgb.XGBClassifier()
model.load_model(XGB_MODEL_PATH)

y_pred = model.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

tt = [[0.31568655, 0.46827707, 0.29714322, 0.48566818, 0.29809314, 0.44381183, 0.30987516, 0.49506208, 0.31187564,
       0.41192672, 0.39518425, 0.53773987, 0.40750334, 0.37108013, 0.54864001, 0.54485959, 0.57668674, 0.36495280,
       0.40422648, 0.52696562, 0.66012847, 0.45730060, 0.62978804, 0.52387440, 0.64198363, 0.41571534, 0.63253820,
       0.69183034, 0.64137560, 0.31220114, 0.69700688, 0.47114739, 0.71497583, 0.52870202
       ]]
# tt = 1

print(model.best_ntree_limit)
print(model.predict(tt))


