# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 10:47:23 2015

@author: Yiji
"""

import numpy as np
import util
import sys
import matplotlib.pyplot as plt

#==============================================================================
# Global variables
#==============================================================================


#==============================================================================
# Understand feature types
#==============================================================================

#### Yiji: a buggy implementation though....=.= could be potentially buggy..
def getFeatures(data):
    numRows = int(data.shape[0])
    numCols = int(data.shape[1])  # total number of features = # of columns -1 (because the last column is "target")
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
    #print "%s The feature types are:" % util.RESULT
    #print "%s \n" % featureType
    
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

# key: the number of distinct values
# value: a set of features that have "key" number of distinct values
distinctValCntFeatureMap = dict()

def getValueCountsAll(data):
    print "%s Value counts: " % util.RESULT
    featureSize = int(data.shape[1])  # total number of features
#    print "Feature size is: %d \n" % featureSize
    for i in range(featureSize): # for values of each feature. 
        col = i
        colArray = data[:,col]
        
        d = getValueCounts(colArray)
        featureSet = set()
        if len(d) in distinctValCntFeatureMap:
            featureSet = distinctValCntFeatureMap.get(len(d))
        featureSet.add(i+1) # +1 because the Feature ID starts from 1 but not 0
        distinctValCntFeatureMap[len(d)] = featureSet
        
        
        
        # get the number of values for this feature.
        print "Feature %d : " % i
        for keys, values in d.items():
            print(keys,values)

#==============================================================================
# Understand "target" column
#==============================================================================  
def countTargetValues(dataFile):
     data = util.loadCsv(dataFile)
     targets = data[:,-1]     
     targets = map(int, targets)
     print np.count_nonzero(np.asarray(targets))
     
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
    data = np.asarray(data)[:,:-1]
#    print data
    
    ### Get and output feature types
    analayzeFeatureType(data)
    
    ### Get and output value counts for each feature
    getValueCountsAll(data)
    
    print "\n%s The number of distinct values and the corresponding feature IDs." % util.RESULT
    print distinctValCntFeatureMap
    
    distinctValCntFeatureCntMap = dict()
    for keys, values in distinctValCntFeatureMap.items():
        distinctValCntFeatureCntMap[keys] = len(values)
        
    print "\n%s The number of distinct values and # of features." % util.RESULT
    print distinctValCntFeatureCntMap
    
    # plot
    x = [0] * len(distinctValCntFeatureCntMap)
    singley = [0] * len(distinctValCntFeatureCntMap)
    accumy = [0] * len(distinctValCntFeatureCntMap)
    
    accumValue = 0
    for keys, values in distinctValCntFeatureCntMap.items():
        x.append(keys)
        singley.append(values)
        accumValue += values
        accumy.append(accumValue)
        
    plt.plot(x, singley)
    plt.xlabel('# of distinct values in a feature')
    plt.ylabel('# of features')
    plt.show()
    
    plt.plot(x, accumy)
    plt.xlabel('# of distinct values in a feature')
    plt.ylabel('accumalated # of features')
    plt.show()

    ### redirecting stdout    
    sys.stdout = orig_stdout
    f.close()

    
main()
#countTargetValues('train.csv')

