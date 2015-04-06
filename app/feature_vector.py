import sys
import os
import json
import time
import string
import operator

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if not path in sys.path:
    sys.path.insert(1, path)
del path

try:
    import database.mongo as mongo
except ImportError as exc:
    print("Error: failed to import settings module ({})".format(exc))

try:
    import nltk
except ImportError as exc:
    print("Error: failed to import settings module ({})".format(exc))

try:
	from bs4 import BeautifulSoup
except ImportError as exc:
	print("Error: failed to import settings module ({})".format(exc))

try:
	from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError as exc:
	print("Error: failed to import settings module ({})".format(exc))

from sklearn.feature_extraction.text import TfidfVectorizer	

import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MultiLabelBinarizer

from app import stat
from time import time

test_data = "data/processed.csv"
takeword_data = "data/take_word.txt"
replaceword_data = "data/replaceword.txt"


def get_featureVector(input_size = 100000, select_transform = 1, read_database = 1):

	to_print = 0
	train = stat.get_featurematrix(input_size, select_transform = select_transform, read_database = read_database, to_print = to_print)
	count = 0
	# print b.shape
	t0 = time()
	U, s, V = np.linalg.svd(train, full_matrices=True)
	print("SVD decomposition done in %fs" % (time() - t0))
	square_sum_s = np.square(s).sum()
	#not sure if this is the most optimal way for finding the sum of squares

	# print "squared sum = "+str(square_sum_s)
	temp_sum = 0
	count = 0
	for i in s:
		temp_sum+= i*i
		count+=1 
		if(temp_sum >= 0.9*square_sum_s):
			break;
	# print count
	# print s.shape
	# print V.shape
	processedV = np.transpose(np.delete(V, np.s_[count::1], 0))
	#can use splicing instead of delete
	print processedV.shape
	train_data = np.dot(train, processedV)
#	print b.shape
#	print processedV.shape
	print train_data.shape
	
	train_results = stat.get_trainmatrix(input_size, read_database = read_database, to_print = to_print)

	mlb = MultiLabelBinarizer()
	Y = mlb.fit_transform(train_results)
	#print train_results.shape
	print OneVsRestClassifier(LinearSVC(random_state=0)).fit(train_data, Y).predict(train_data)
#	print s
#	print V

if __name__ == "__main__":
	get_featureVector(200, select_transform = 2, read_database = 0)
