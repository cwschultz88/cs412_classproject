# -*- coding: utf-8 -*-
import csv
import random
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from preprocess import Preprocess
import time

def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]
    
def main():
    print '[INFO, time: %s] Getting Data....' % (time.strftime('%H:%M:%S'))
    preprocesser = Preprocess()
    data= preprocesser.read()

    splitRatio = 0.67
    train, test = splitDataset(data, splitRatio)
    train = np.asarray(train)
    test = np.asarray(test)

    trainX = train[:,:-1]
    trainy = train[:,-1]

    testX = test[:,:-1]
    testy = test[:,-1]

    print '[INFO, time: %s] Fitting %s ...' % (time.strftime('%H:%M:%S'), 'Gradient Boosting Classifier with 300 estimators')
    clf = GradientBoostingClassifier(n_estimators=300)
    clf.fit(trainX, trainy)

    print '[INFO, time: %s] Making Predictions...' % (time.strftime('%H:%M:%S'))
    prediction = clf.predict(testX)
    print '[RESULT, time: %s] accuracy = %f' % (time.strftime('%H:%M:%S'),accuracy_score(testy, prediction))

main()






