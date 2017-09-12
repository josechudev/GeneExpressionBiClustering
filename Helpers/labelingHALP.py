import re
import string
import pandas as pd
import numpy as np
import gc

class LabelHelper:
	def __init__(self,files):
		filenames = files
		
	def getFilenames(self):
		return filenames

	def processFiles(self):

		for f in files:
			df = pd.read_csv(f,sep = ';')
			df['TargetLabel'] = np.nan
			i=0
			for index,row in df.iterrows():
				if (row['Pvalue']<=0.01):
					df.loc[i,'TargetLabel'] =1
				elif (row['Pvalue']>=0.5):
					df.loc[i,'TargetLabel'] =0
				i+=1
			lstfilename = f.split('.')
			filename = "../Labeled" + lstfilename[-2] + 'Labeled.' + lstfilename[-1]
			print ("File created: " + filename + "\n")
			df.to_csv(filename, sep = ';')
			del df
			gc.collect()


