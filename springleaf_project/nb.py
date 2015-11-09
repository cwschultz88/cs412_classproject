# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 09:38:59 2015

Referece:
1) multinomial Naive Bayes classifier
http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html


@author: Yiji
"""

import csv
import random
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import operator
#==============================================================================
# Global variables
#==============================================================================
INFO = '[INFO]'
DEBUG = '[DEBUG]'
RESULT= '[RESULT]'

def loadCsv(fname):
    lines = csv.reader(open(fname, "rU"),delimiter=",")
    data = list(lines)
    dataarray = np.asarray(data)[1:,1:] # skip the first line (the feature name), and the first column (IDs)
    
    return dataarray
    
def preprocessData(data):
    featureSize = int(data.shape[1])  # total number of features
    newData = np.zeros((len(data),featureSize))
#    print newData
    
    for i in range(featureSize): # for values of each feature. 
        col = i - 1
        colArray = data[:,col]
        d = dict() # to hold {string,intID} map
        row = 0
        
#==============================================================================
#        numID:
#        used to map an int identifier to each distinct string value; 
#        also used for missing data, but this is not well designed because the placeholder for missing data 
#        should not be the same as any existing values of the same feature. We'll deal with it later.        
#==============================================================================
        numId = 0
        for element in colArray:
            if isInt(element):
                element = int(element)
                # for negative data:                 
                if element < 0:
                    element = abs(element)
                newData[row,col] = element
                #print element
            elif isLong(element):
                element = long(element)
                newData[row,col] = element
            elif isFloat(element):
                element = float(element)
                newData[row,col] = element
            else: #if the value is string, then replace string with int value.
#                print element
#                print d     
#                print ''
                if element not in d:
                    d[element]=numId
                    numId += 1
                element = d[element]
                newData[row,col] = element
            row += 1
                
    return newData    
#==============================================================================
# take an array, iterate through it and count the occurrences
# return a dictionary of {value, Count} 
#==============================================================================
def countOccurrences(array):
    d = dict()
    for key in array:
        # check and/or convert value type of key
        if isInt(key):
            key = int(key)
        elif isLong(key):
            key = long(key)
        elif isFloat(key):
            key = float(key)
#        else:
#            #string
        
        # update count map for each key
        if key in d:
            d[key]+=1
        else:
            d[key]=1
#    print d
    return d
    # if the key is a string
    

def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated


#==============================================================================
# Helper functions to check variable type
#==============================================================================   
def isInt(valueStr):    
    try: 
        int(valueStr)
        return True
    except ValueError:
        return False

def isLong(valueStr):
    try: 
        long (valueStr)
        return True
    except ValueError:
        return False

def isFloat(valueStr):
    try:
        float(valueStr)
        return True
    except ValueError:
        return False


def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]


featureAccuracy = dict()

def eachFeatureMain():
    dataFile = 'example.csv'

    print '%s Loading and pre-processing data...' % INFO 
    data = loadCsv(dataFile)
#    print '%s Original training data:' % DEBUG
    data = preprocessData(data)
#    print '%s Pre-processed training data:' % DEBUG
#    print data

    # Split the data into training and testing    
    splitRatio = 0.80
    train, test = splitDataset(data, splitRatio)
    train = np.asarray(train)
    test = np.asarray(test)

    trainy = train[:,-1]
    testy = test[:,-1]
    
    print '%s Feature 1...' % (INFO)
    trainX = train[:,:1]
    testX = test[:,:1]
        
    print '    %s Fitting Naive Bayes classifier...' % INFO 
    clf = MultinomialNB().fit(trainX, trainy)
    print '    %s Making predictions...' % INFO 
        # for testing
    #    prediction = clf.predict(trainX)
    prediction = clf.predict(testX)
    #print '    [RESULT] the prediction is: '    
    #print prediction
    print '    %s Calcualting prediction accuracy...' % INFO 
        # for testing
    #    accuracy = np.mean(prediction == trainy) 
    accuracy = np.mean(prediction == testy)  
    print '    [RESULT] accuracy = %f' % accuracy    
    featureAccuracy[0] = accuracy
    
    
    featureSize = int(data.shape[1])-1
    # Iterate through each feature
    for i in range(1,featureSize):
        print '%s Feature %d...' % (INFO, i+1)
        trainX = train[:,i:i+1]
        testX = test[:,i:i+1]
        
        print '    %s Fitting Naive Bayes classifier...' % INFO 
        clf = MultinomialNB().fit(trainX, trainy)
        print '    %s Making predictions...' % INFO 
        # for testing
    #    prediction = clf.predict(trainX)
        prediction = clf.predict(testX)
        #print '    [RESULT] the prediction is: '    
        #print prediction
        print '    %s Calcualting prediction accuracy...' % INFO 
        # for testing
    #    accuracy = np.mean(prediction == trainy) 
        accuracy = np.mean(prediction == testy)  
        print '    [RESULT] accuracy = %f' % accuracy
        featureAccuracy[i] = accuracy
    
    #print featureAccuracy
    # sort by accuracy
    sorted_featureAccuracy = sorted(featureAccuracy.items(), key=operator.itemgetter(1),reverse=True)

    print ''
    print '%s Final sorted result in format (FeatureID-1, Accuracy):' % RESULT
    print sorted_featureAccuracy
    
def main():
#    trainFile = 'train.csv'    
#    testFile = 'test.csv'
    dataFile = 'example.train.csv'


#    print '%s Loading and pre-processing training data...' % INFO 
#    trainData = loadCsv(trainFile)
##    print '%s Original training data:' % DEBUG
##    print trainData
#    trainData = preprocessData(trainData)
##    print '%s Pre-processed training data:' % DEBUG
##    print trainData
#    
#    print '%s Loading and pre-processing testing data...' % INFO 
#    testData = loadCsv(testFile)
#    testData = preprocessData(testData)
##    print testData
#    

    
    print '%s Loading and pre-processing data...' % INFO 
    data = loadCsv(dataFile)
#    print '%s Original training data:' % DEBUG
    data = preprocessData(data)
#    print '%s Pre-processed training data:' % DEBUG
#    print data

    # Split the data into training and testing    
    splitRatio = 0.67
    train, test = splitDataset(data, splitRatio)
    train = np.asarray(train)
    test = np.asarray(test)
#    print train
#    print test
    
    trainX = train[:,:-1]
    trainy = train[:,-1]
    
    testX = test[:,:-1]
    testy = test[:,-1]
    
    print '%s Fitting Naive Bayes classifier...' % INFO 
    clf = MultinomialNB().fit(trainX, trainy)
    print '%s Making predictions...' % INFO 
    # for testing
#    prediction = clf.predict(trainX)
    prediction = clf.predict(testX)
    print '[RESULT] the prediction is: '    
    print prediction
    print '%s Calcualting prediction accuracy...' % INFO 
    # for testing
#    accuracy = np.mean(prediction == trainy) 
    accuracy = np.mean(prediction == testy)  
    print '[RESULT] accuracy = %f' % accuracy
#==============================================================================
#     the code below is for the purpose of manually creating Naive Bayes clf
#==============================================================================
#    featureSize = int(trainData.shape[1])  # total number of features
#    
#    occArray = [] # each element is a dictionary of {value,count} for each feature
#    for i in range(featureSize):
#        col = i - 1
#        colArray = trainData[:,col]
#        occArray.append(countOccurrences(colArray))
#        #print trainData[:,col]
#    print occArray
#    separated = separateByClass(trainData)
#    print('Separated instances: {0}').format(separated)
    
#    print dataarray[:,0] #access the first column
    
    
#main()
eachFeatureMain()

#==============================================================================
# Testing code
#==============================================================================
#print isInt('-100')
#print isFloat('-3.14567')



