import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from numpy.random import RandomState


def split_dataset(df, validation_percentage, seed):
    state = RandomState(seed)
    validation_indexes = state.choice(df.index, int(len(df.index) * validation_percentage), replace=False)
    training_set = df.loc[~df.index.isin(validation_indexes)]
    validation_set = df.loc[df.index.isin(validation_indexes)]
    return training_set, validation_set


def train(datafile):
    data = pd.read_parquet(datafile)
    training_set, validation_set = split_dataset(data, 0.25, 1)

    #train
    training_set["s1"] = training_set["s1"].fillna(training_set["s1"].mean())
    training_set["default"] = training_set["default"].fillna(training_set["default"].mode()[0])
    clf = LogisticRegression(C=0.1)
    clf.fit(training_set[["s1", "s2", "s3", "s4"]], training_set["default"])

    # evaluation
    validation_set["s1"] = validation_set["s1"].fillna(training_set["s1"].mean())
    validation_set["default"] = validation_set["default"].fillna(training_set["default"].mode()[0])
    validation_predictions = clf.predict_proba(validation_set[["s1", "s2", "s3", "s4"]])[:, 1]
    auc = roc_auc_score(validation_set[["default"]], validation_predictions)

    return clf, auc
