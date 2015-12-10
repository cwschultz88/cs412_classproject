# -*- coding: utf-8 -*-
import csv
import random
import numpy as np
import pickle
from preprocess import Preprocess
import re
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sknn.mlp import Classifier, Layer
from sknn.ae import Layer, AutoEncoder
from sklearn.metrics import accuracy_score
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

    print '[INFO, time: %s] Fitting %s ...' % (time.strftime('%H:%M:%S'), 'Auto Encoder')
    ae = AutoEncoder(
    	layers=[
		Layer("Sigmoid",units=100)
               ],
	learning_rate=0.01,
	n_iter=40,
        verbose=True,
    )
    ae.fit(data[:,:-1])

    print '[INFO, time: %s] Transforming Data with %s ...' % (time.strftime('%H:%M:%S'), 'Auto Encoder')
    splitRatio = 0.67
    train, test = splitDataset(data, splitRatio)
    train = np.asarray(train)
    test = np.asarray(test)
    
    trainX = train[:,:-1]
    trainy = train[:,-1]
    
    testX = test[:,:-1]
    testy = test[:,-1]

    transformed_trainX = ae.transform(trainX)
    transformed_testX = ae.transform(testX)

    print '[INFO, time: %s] Fitting %s ...' % (time.strftime('%H:%M:%S'), 'SVM - rbf kernel (i.e. gaussian) with default paramenters')
    clf = SVC()
    clf.fit(transformed_trainX, trainy)

    print '[INFO, time: %s] Making Predictions...' % (time.strftime('%H:%M:%S'))
    prediction = clf.predict(transformed_testX)
    print '[RESULT, time: %s] accuracy = %f' % (time.strftime('%H:%M:%S'),accuracy_score(testy, prediction))

main()






