import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import cross_val_score, KFold

voice = pd.read_csv("voice.csv")

leafSizeList = [1,3,5,10,15,20]
depthList = [3,5,8,10,15]
ccr = pd.DataFrame(({"leafSize":[],"depth":[],"score":[]}))

kfold = KFold(n_splits=10,random_state=1)
for leafSize in leafSizeList:
    for depth in depthList:
        ctree = tree.DecisionTreeClassifier(min_samples_leaf=leafSize, max_depth=depth, random_state=1)
        ctree.fit(voice.drop("label", axis=1), voice.label)
        scores = cross_val_score(ctree, X=voice.drop("label", axis=1), y=voice["label"], cv=kfold)
        ccr = ccr.append(pd.Series((leafSize, depth, scores.mean()), index=["leafSize", "depth", "score"]), ignore_index=True)
        print(str(leafSize)+" "+str(depth)+" finished")

print(ccr)
print(ccr[ccr.score==ccr.score.max()])

# best tree is depth =3. the minimal leaf size could be 5, 10, 15, 20. the best ccr is 0.9583