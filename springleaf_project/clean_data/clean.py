import numpy as py
import pandas as pd
from pandas import DataFrame, Series

df = pd.read_csv('test.csv')
df = df.drop(df.columns[0], axis=1)
df = df.loc[:,df.max(axis=0) != df.min(axis=0)]
df = df.fillna(-1)
df.to_csv('data_cleaned.csv', sep=',',index=False)