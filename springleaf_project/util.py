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