import pandas as pd

class Preprocess():

    def read(self):
		# values based on observation, still need to find a way to get them programatically
		na = ["NA","NAN","-1","-99999","999999990","999999991","999999992","999999993","999999994","999999995","999999996","999999997","999999998","999999999"]

		df = pd.read_csv('test.csv',na_values=na)
		df = df.drop(df.columns[0], axis=1)
		df = df.loc[:,df.max(axis=0) != df.min(axis=0)]
		df = df.fillna(-1)

		#df.to_csv('data_cleaned.csv', sep=',',index=False)
		from datacleanerShiwangi import DataCleanerShiwangi
		return DataCleanerShiwangi.clean(df.as_matrix())