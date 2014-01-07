__author__ = 'CJank'

import warnings

import sklearn as skl
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

def crossValidManyClassifiers(classifiers, X, y, cv_param=10):
    listToReturn=[]
    for clf in classifiers:
        scores = skl.cross_validation.cross_val_score(clf, X, y, cv=cv_param)
        CVscore = np.mean(scores)

        #print name+"-> "+str(score)
        #print name+"-> "+str(score_train)+" (on training set)"
        listToReturn.append(CVscore)

    return listToReturn

class CombinedClassifier(BaseEstimator,  ClassifierMixin):

    def __init__(self,classifiers):
        self.classifiers = classifiers

    def fit(self,X,y):
        for clf in self.classifiers:
            clf.fit(X,y)

        return self

    def predict(self,X):
        pred=[]
        for clf in self.classifiers:
            pred.append(clf.predict(X))

        #predArray=np.array(pred).reshape(self.classifiers.__len__(),pred[0].__len__())

        numbersOfDecisions = np.zeros((pred[0].__len__(),int(np.array(pred).max())+1))
        pointIndex=0
        for clfD in pred:
            pointIndex=0
            for point in clfD:
                numbersOfDecisions[pointIndex,int(point)]+=1
                pointIndex+=1


        decArray= numbersOfDecisions.argmax(axis=1)
        return decArray.tolist()