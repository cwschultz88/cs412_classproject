# -*- coding: utf-8 -*-
import csv
import random
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from multiprocessing import Process, Pool

import time

def f(number, trainX, testX, trainy, testy):
    clf = SVC()

    clf.fit(trainX, trainy)

    prediction = clf.predict(testX)
  
    print '[RESULT, time: %s] Feature %d accuracy = %f\n' % (time.strftime('%H:%M:%S'), number ,accuracy_score(testy, prediction))

def chunks(l, n):
    for i in xrange(0, len(l), n):
	yield l[i:i+n]
   
def main():
    number_of_sub_processes = 16

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

    print '[INFO, time: %s] Fitting SVM - rbf kernel (i.e.) gaussian with default parameters\n' % (time.strftime('%H:%M:%S'))

    for chunk in chunks(range((train.shape[1] - 1)), number_of_sub_processes):
      processes = []

      for index in chunk:
        trainX = np.array([[x] for x in train[:, index]])
        testX = np.array([[x] for x in test[:, index]])
        
        processes.append(Process(target=f, args=(index + 1, trainX, testX, trainy, testy)))

      for p in processes:
        p.start()

      for p in processes:
        p.join()

main()






