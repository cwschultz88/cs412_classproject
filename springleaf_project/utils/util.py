# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 10:48:38 2015

@author: Yiji
"""
import csv
import numpy as np

#==============================================================================
# Global variables
#==============================================================================
INFO = '[INFO]'
DEBUG = '[DEBUG]'
RESULT = '[RESULT]'

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

def getDataType(element):
    if isInt(element):
        fType = "int"
    elif isFloat(element):
        fType = "float"
    elif isLong(element):
        fType = "long"
    else:
        fType = "string"
    return fType