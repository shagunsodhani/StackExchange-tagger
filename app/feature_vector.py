import sys
import os
import json
import string
import operator
from time import time

import numpy as np
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import MultiLabelBinarizer

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if not path in sys.path:
    sys.path.insert(1, path)
del path

try:
    import database.mongo as mongo
except ImportError as exc:
    print("Error: failed to import settings module ({})".format(exc))

try:
	from app import stat
except ImportError as exc:
    print("Error: failed to import settings module ({})".format(exc))


def accuracy_atleast_one_match(result, prediction):
	length = len(result)
	#print length
	count = 0.0
	for i in range(0, length):
		flag = 0
		for j in prediction[i]:
			if j in result[i]:
				flag = 1
		count+=flag
	print "Accuracy = "+str(count/length)

def accuracy_null_results(result, prediction):
	length = len(result)
	count = 0.0
	for i in range(0, length):
		if not prediction[i]:
			count+=1
	print "null_results = "+str(count/length)

def input_representation(result):
	tag_count = {}
	for i in result:
		for j in i:
			if j not in tag_count:
				tag_count[j] = 1
			else:
				tag_count[j]+=1
	for j in tag_count:
		print str(j)+" : "+str(tag_count[j])

def get_featureVector(input_size = 100000, select_transform = 1, read_database = 1):

	to_print = 0
	train = stat.get_featurematrix(input_size, select_transform = select_transform, read_database = read_database, to_print = to_print)
	t0 = time()
	U, s, V = np.linalg.svd(train, full_matrices=True)
	print("SVD decomposition done in %fs" % (time() - t0))
	square_sum_s = np.square(s).sum()
	#not sure if this is the most optimal way for finding the sum of squares

	temp_sum = 0
	count = 0
	for i in s:
		temp_sum+= i*i
		count+=1 
		if(temp_sum >= 0.9*square_sum_s):
			break;

	processedV = np.transpose(np.delete(V, np.s_[count::1], 0))
	#can use splicing instead of delete
	train_data = np.dot(train, processedV)	
	train_results = stat.get_trainmatrix(input_size, read_database = read_database, to_print = to_print)
	
	mlb = MultiLabelBinarizer()
	Y = mlb.fit_transform(train_results)
	print type(Y)
	#print train_results.shape
	#class sklearn.svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000
	clf = OneVsRestClassifier(svm.LinearSVC(random_state=0, dual = False, max_iter =20000, verbose = 0))

	# clf = OneVsOneClassifier(LinearSVC(random_state=0, max_iter =20000, verbose = 0))
	#clf = OneVsOneClassifier(SVC(random_state=0, verbose = 0))
	prediction_Y  = clf.fit(train_data, Y).predict(train_data)

#	predict_Y

	#class sklearn.svm.SVC(C=1.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)
	#prediction_Y = OneVsRestClassifier(SVC(random_state=0, verbose = 0)).fit(train_data, Y).predict(train_data)
	#print type(prediction_Y)
	prediction = mlb.inverse_transform(prediction_Y)
	for i in prediction:
		print i
	print "\n"
	for i in train_results:
		print i
	print clf.decision_function(train_data)
	print Y


if __name__ == "__main__":
	get_featureVector(10, select_transform = 2, read_database = 1)
