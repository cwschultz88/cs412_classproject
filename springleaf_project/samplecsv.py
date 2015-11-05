# -*- coding: utf-8 -*-
"""
Created on Wed Nov 04 10:57:24 2015

@author: Yiji
"""

import pandas as pd
import numpy as np

def main():
    in_filename = 'train.csv'
    out_filename = 'sample.csv'
    nlinesfile = 145232
    nlinesrandomsample = 10001
#    nlinesfile = 6
#    nlinesrandomsample = 3
    lines2skip = np.random.choice(np.arange(1,nlinesfile+1), (nlinesfile-nlinesrandomsample), replace=False)
    df = pd.read_csv(in_filename, skiprows=lines2skip,low_memory=False)
    df.to_csv(out_filename, sep=',',index=False)
    

main()