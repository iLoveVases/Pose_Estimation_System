import xgboost as xgb
import pandas as pd
import sklearn as sk
from CreateDataBase import CSV_generate
from sklearn import metrics
from sklearn.metrics import classification_report
from definitions import XGB_MODEL_PATH, CSV_PATH


dana = CSV_generate.data
# print(np.size(dana))
classes = ['sitting', 'standing', 'lying']
df = pd.read_csv(CSV_PATH + "/data.csv")

X = df.drop(' POSE', axis=1).copy().values  # spacje popraw potem
y = df[' POSE'].copy().values

X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size=0.2)

clf_xgb = xgb.XGBClassifier(objective='binary:logistic', gamma=0.25, learn_rate=0.1, max_depth=4, seed=42)

model = clf_xgb.fit(X_train,
                    y_train,
                    verbose=True,
                    early_stopping_rounds=10,
                    eval_metric='mlogloss',
                    eval_set=[(X_test, y_test)])

y_pred = model.predict(X_test)


print("F1 macro score: ", metrics.f1_score(y_test, y_pred, average='macro'))
print("F1 micro score: ", metrics.f1_score(y_test, y_pred, average='micro'))
print("F1 weighted score: ", metrics.f1_score(y_test, y_pred, average='weighted'))
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred, target_names=classes))

model.save_model(XGB_MODEL_PATH)
