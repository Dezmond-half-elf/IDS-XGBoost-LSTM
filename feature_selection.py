from xgboost import XGBClassifier
import numpy as np

def xgb_create(x, y, estimators):
    xgb = XGBClassifier(booster='gbtree', max_depth=10, learning_rate=0.01, n_estimators=estimators)
    xgb.fit(x, y)

    return xgb


def xgb_accuracy_count(xgb, x_train, x_test, y_train, y_test):
    predict_train = xgb.predict(x_train)
    predict_test = xgb.predict(x_test)
    correct_train_predictions = np.sum(predict_train == y_train)
    correct_test_predictions = np.sum(predict_test == y_test)

    return correct_train_predictions/predict_train.shape[0], correct_test_predictions/predict_test.shape[0]


def feature_select(n_features, xgb, x_train, x_test):
    FI = xgb.feature_importances_
    sorted_idx = FI.argsort()
    best_features = x_train.columns[sorted_idx[-n_features:]]

    return x_train.copy()[best_features], x_test.copy()[best_features]
