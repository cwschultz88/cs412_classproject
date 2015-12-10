# -*- coding: utf-8 -*-
import csv
import random
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import time
    
def main():
    print '[INFO, time: %s] Getting Data....' % (time.strftime('%H:%M:%S'))
    testing_file = file('test.p', 'r')
    training_file = file('train.p', 'r')

    train = pickle.load(training_file)
    test = pickle.load(testing_file)

    testing_file.close()
    training_file.close()
    
    trainX = train[:,:-1]
    trainy = train[:,-1]
    
    testX = test[:,:-1]
    testy = test[:,-1]

    print '[INFO, time: %s] Fitting %s ...' % (time.strftime('%H:%M:%S'), 'SVM - poly kernel degree 25 with default paramenters')
    clf = SVC(kernel='poly', degree=25)
    clf.fit(trainX, trainy)

    print '[INFO, time: %s] Making Predictions...' % (time.strftime('%H:%M:%S'))
    prediction = clf.predict(testX)
    print '[RESULT, time: %s] accuracy = %f' % (time.strftime('%H:%M:%S'),accuracy_score(testy, prediction))

main()
