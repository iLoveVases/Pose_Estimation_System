# import matplotlib.pyplot as plt
# from sklearn import svm
# import numpy as np
# import os
# import pandas as pd
# import sklearn as sk
# #import CSV_generate
# from sklearn import metrics
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import PrecisionRecallDisplay
# from sklearn.metrics import classification_report
#
# #dana = CSV_generate.data
# # print(np.size(dana))
# classes = ['sitting', 'standing', 'lying']
# df = pd.read_csv("data.csv")
#
# X = df.drop(' POSE', axis=1).copy().values  # spacje popraw potem
# y = df[' POSE'].copy().values
#
# # clf_xgb = xgb.XGBClassifier(base_score=0.5, booster='gbtree',
# #                             colsample_bytree=1,
#                             # lcarning_ratc=0.1, nax_dclta_stcp=0, nax_dcpth=3,
#                             # nin_child_weight=1, missing=None, n_estimators=169, n_jobs=1,
#                             # nthread=None, objective='binary:logistic', random_state=0,
#                             # reg_alpha=0, rcg_lanbda=1, scale_pos_weight=1, seed=42,
#                             # silent=None, subsanple=1, verbosity=1, use_label_encoder=False)
#
# X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size=0.2)
#
# # clf_xgb = xgb.XGBClassifier(objective='binary:logistic', gamma=0.25, learn_rate=0.1, max_depth=4, seed=42)
#
# # model = clf_xgb.fit(X_train,
# #                     y_train,
# #                     verbose=True,
# #                     early_stopping_rounds=10,
# #                     eval_metric='aucpr',
# #                     eval_set=[(X_test, y_test)])
# print(np.size(X_train[:, 0]))
# plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='winter')
# #plt.show()
#
# # model_SVM.save_model("SVM_000.json")
# svm = svm.SVC(kernel='polynomial', degree=4, coef0=0.1)
# model_SVM = svm.fit(X_train, y_train)
#
# y_pred = model_SVM.predict(X_test)
#
# print("--------------------------------------------------")
# print("F1 macro score: ", metrics.f1_score(y_test, y_pred, average='macro'))
# print("F1 micro score: ", metrics.f1_score(y_test, y_pred, average='micro'))
# print("F1 weighted score: ", metrics.f1_score(y_test, y_pred, average='weighted'))
# print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
# print("--------------------------------------------------")
#
# # print("Confusion_matrix: ")
# # print(confusion_matrix(y_test, y_pred))
#
#
#
# # display = sk.metrics.PrecisionRecallDisplay.from_estimator( model_SVM, X_test, y_test, name="LinearSVC")
# # _ = display.ax_.set_title("2-class Precision-Recall curve")
#
#
#
# print(classification_report(y_test, y_pred, target_names=classes))
#
#
#
#
# #model_SVM.save_model("SVM_200.json")
#
#
#
# # tt = [[0.31568655, 0.46827707, 0.29714322, 0.48566818, 0.29809314, 0.44381183, 0.30987516, 0.49506208, 0.31187564,
# #        0.41192672, 0.39518425, 0.53773987, 0.40750334, 0.37108013, 0.54864001, 0.54485959, 0.57668674, 0.36495280,
# #        0.40422648, 0.52696562, 0.66012847, 0.45730060, 0.62978804, 0.52387440, 0.64198363, 0.41571534, 0.63253820,
# #        0.69183034, 0.64137560, 0.31220114, 0.69700688, 0.47114739, 0.71497583, 0.52870202
# #        ]]
# # print(model_SVM.best_ntree_limit)
# # print(model_SVM.predict(tt))
