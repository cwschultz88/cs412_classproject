# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 10:47:23 2015

@author: Yiji
"""

import numpy as np
import util
import sys

#==============================================================================
# Global variables
#==============================================================================


#==============================================================================
# Understand feature types
#==============================================================================
def getFeatures(data):
    numRows = int(data.shape[0])
    numCols = int(data.shape[1])  # total number of features; # of columns
    print "Number of rows (entries):     %d" % numRows
    print "Number of columns (features): %d\n" % numCols    
    
    featureType = [0] * numCols
    missingDataColIdx = []
    for colIdx, element in enumerate(data[0]):
#        print element, colIdx
        if element:    # if the string is NOT empty
            featureType[colIdx] = util.getDataType(element)
        else: # record the colIdx of the missing data, which will be used to check in the next ro
            missingDataColIdx.append(colIdx)
        
#        print featureType
#        print missingDataColIdx
   
    for missingColIdx in missingDataColIdx:
        rowIdx = 1
        while(not data[rowIdx,missingColIdx]):
            rowIdx += 1
        featureType[missingColIdx] = util.getDataType(data[rowIdx,missingColIdx])
#        print featureType
   
    return featureType


def analayzeFeatureType(data):
    featureType = getFeatures(data)
    print "%s The feature types are:" % util.RESULT
    print "%s \n" % featureType
    
    print "%s Feature types counts: " % util.RESULT
    print np.unique(featureType, return_counts=True)
    print ""
    
#==============================================================================
# Understand each single feature
#==============================================================================
def getValueCounts(array):
    d = dict()
    for key in array:
       if key in d:
           d[key]+=1
       else:
           d[key]=1
    return d

def getValueCountsAll(data):
    print "%s Value counts: " % util.RESULT
    featureSize = int(data.shape[1])  # total number of features
#    print "Feature size is: %d \n" % featureSize
    for i in range(featureSize): # for values of each feature. 
        col = i - 1
        colArray = data[:,col]
        d = getValueCounts(colArray)
        print "Feature %d : " % i
        for keys, values in d.items():
            print(keys,values)

    
    
#==============================================================================
# Main
#==============================================================================
def main():
    ### redirecting stdout    
    orig_stdout = sys.stdout
    f = file('analyszedata.out', 'w')
    sys.stdout = f
    
    ### Load file
    dataFile = 'example.train.csv'    
    data = util.loadCsv(dataFile)
#    print data
    
    ### Get and output feature types
    analayzeFeatureType(data)
    
    ### Get and output value counts for each feature
    getValueCountsAll(data)
    
    
    

    ### redirecting stdout    
    sys.stdout = orig_stdout
    f.close()
    
main()












