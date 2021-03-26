import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from splearn.EnsembleLearning.EnsembleLearning import AdaBooster
from splearn.DecisionTree.DecisionTree import DecisionTree
from sklearn.metrics import confusion_matrix
import splearn.Metrics as spmet
import sklearn.metrics as skmet

bank_data: pd.DataFrame = pd.read_csv("data/bank/train.csv")
features = bank_data[bank_data.columns[:-1]]
target = bank_data[bank_data.columns[-1]]

X_train, X_test, y_train, y_test = train_test_split(features, target, train_size = 0.33)

ab = AdaBooster(DecisionTree)
ab.train(
    X_train,
    y_train,
    200,
    learning_rate = 1,
    learner_args={
        "gain": "entropy",
        "max_depth": 2
    }
)
print("Model trained!")

dt = DecisionTree()
dt.train(X_train, y_train, gain="entropy", max_depth = 2)

dt_preds = dt.predict(X_test)
ab_preds = ab.predict(X_test)


print(spmet.accuracy_score(dt_preds, y_test))
print(confusion_matrix(dt_preds, y_test))
print()
print(spmet.accuracy_score(ab_preds, y_test))
print(confusion_matrix(ab_preds, y_test))