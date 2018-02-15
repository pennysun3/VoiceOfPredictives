import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, KFold

voice = pd.read_csv("voice.csv")
voice["gender"]=(voice["label"]=="male")*1


# simple regression
regr = linear_model.LinearRegression()
regr.fit(X=voice.ix[:,0:20],y=voice["gender"])
regr.coef_
regr.intercept_
regr.score(X=voice.ix[:,0:20],y=voice["gender"])


## source coude of output layer activation function
'''
# Output for numeric regression
if not is_classifier(self):
    self.out_activation_ = 'identity'
# Output for multi class
elif self._label_binarizer.y_type_ == 'multiclass':
    self.out_activation_ = 'softmax'
# Output for binary class and multi-label
else:
    self.out_activation_ = 'logistic'
'''

# finding a best combination of hidden nodes number and alpha using 5-fold cross validation
alphaList = [0.001,0.01,0.1,1,10]
numNodesList = [20,30,40,50,60,70,80,90,100]
ccr = pd.DataFrame(({"alpha":[],"numNodes":[],"score":[]}))

kfold = KFold(n_splits=10,random_state=1)
for alpha in alphaList:
    for numNodes in numNodesList:
        nn = MLPClassifier(hidden_layer_sizes=numNodes, activation="logistic", alpha=alpha, solver="lbfgs", random_state=1)
        nn.fit(X=voice.ix[:,0:20],y=voice["gender"])
        scores = cross_val_score(nn, X=voice.ix[:, 0:20], y=voice["gender"], cv=kfold)
        ccr = ccr.append(pd.Series((alpha, numNodes, scores.mean()), index=["alpha", "numNodes", "score"]), ignore_index=True)
        print(str(alpha)+" "+str(numNodes)+" finished")

ccr[ccr.score == ccr.score.max()]
ccr.to_csv("ccr_neural_network",index=False,encoding="utf-8")

# neural network with alpha = 0.01, one hidden layer with 20 hidden nodes, logistic form for both hidden and output layer
nn = MLPClassifier(hidden_layer_sizes=20,activation="logistic",alpha=0.01,solver="lbfgs",random_state=1)
nn.fit(X=voice.ix[:,0:20],y=voice["gender"])
cross_val_score(nn, X=voice.ix[:, 0:20], y=voice["gender"], cv=kfold)

