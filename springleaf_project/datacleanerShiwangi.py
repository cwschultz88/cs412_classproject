import util as ut
import csv 
import pandas as pd
from pandas import DataFrame as df, read_csv, Series
from datacleaner import DataCleaner

class DataCleanerShiwangi(DataCleaner):
    @staticmethod        
    def clean(numpy_array):  #load your csv data here in numpy_array
        data=ut.preprocessData(numpy_array)

        #print dataarray
        #print data

        ###### numpy into pandas dataframe
        df = pd.DataFrame(data)
        #print df
        #print df.dtypes

        df=df.astype('float16')
        #print df.dtypes


        ###### generate preprocessed csv file 
        #df.to_csv('preprocessed_data.csv', sep=',',index=False)

        ###### normalize data between [0,1] using X_norm= (X - Xmin)/ (Xmax - Xmin)
        df_norm= (df - df.min()) / (df.max()-df.min())
        df_norm=df_norm.fillna(-1)

        ##### generate normalized csv 
        #df_norm.to_csv('normalized_data.csv',sep=',', index=False)
        
        return df_norm.as_matrix() 
