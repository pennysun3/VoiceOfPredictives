import pandas as pd
import numpy as np
from sklearn import neighbors
from sklearn.model_selection import cross_val_score, KFold


def standardize(column):
    return (column - np.mean(column))/np.std(column)

voice = pd.read_csv("voice.csv")
label = voice.label
voice = voice.drop(labels = "label", axis = 1)
voice = voice.apply(standardize)
voice["label"] = label

nnList = [5,10,15,20]
weightsList = ["uniform","distance"]
ccr = pd.DataFrame(({"nn":[],"weights":[],"score":[]}))

kfold = KFold(n_splits=10,random_state=1)
for nn in nnList:
    for weights in weightsList:
        knn = neighbors.KNeighborsClassifier(n_neighbors=nn, weights=weights)
        knn.fit(X=voice.ix[:, 0:20], y=voice.label)
        scores = cross_val_score(knn, X=voice.drop("label", axis=1), y=voice["label"], cv=kfold)
        ccr = ccr.append(pd.Series((nn, weights, scores.mean()), index=["nn", "weights", "score"]), ignore_index=True)
        print(str(nn)+" "+str(weights)+" finished")

print(ccr)

# highest score: 0.9369
# from: nn=5, weights = "uniform"
