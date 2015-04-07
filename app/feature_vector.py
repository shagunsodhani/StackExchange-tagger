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

try:
	from app import evaluate
except ImportError as exc:
    print("Error: failed to import settings module ({})".format(exc))

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

def predict(input_size = 100000, select_transform = 1, read_database = 1, one_vs_one = 0, model = "LinearSVC"):

	to_print = 0
	raw_train_data = stat.get_featurematrix(input_size, select_transform = select_transform, read_database = read_database, to_print = to_print)
	t0 = time()
	U, s, V = np.linalg.svd(raw_train_data, full_matrices=True)
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

	print "count = "+str(count)
	x = np.delete(V, np.s_[count::1], 0)
	processedV = np.transpose(x)
	X = np.dot(raw_train_data, processedV)
	
	# print "count = "+str(count)
	# print "V.shape = "+str(V.shape)
	# print "s.shape = "+str(s.shape)
	# x = np.delete(V, np.s_[count::1], 0)
	# print "x.shape = "+str(x.shape)
	# print "raw_train_data.shape = "+str(raw_train_data)
	# print "processedV.shape = "+str(processedV.shape)

	#can use splicing instead of delete
	
	# print "X.shape = "+str(X.shape)	

	train_results = stat.get_trainmatrix(input_size, read_database = read_database, to_print = to_print)
	
	mlb = MultiLabelBinarizer()
	Y = mlb.fit_transform(train_results)

	# print Y.shape

	if(one_vs_one == 1):
		clf = OneVsOneClassifier(svm.LinearSVC(random_state=0, max_iter =10000, verbose = 0))
		prediction_Y  = clf.fit(X, Y).predict(X)
	else:
		if model == "LinearSVC":
			print "Showing Results for one vs rest multilable classifier using LinearSVC model"
			clf = OneVsRestClassifier(svm.LinearSVC(random_state=0, dual = False, max_iter =10000, verbose = 0))
			
		elif model == "SVC":
			print "Showing Results for one vs rest multilable classifier using SVC model"
			clf = OneVsRestClassifier(svm.SVC(verbose = 0))
		scores = clf.fit(X, Y).decision_function(X)
			# print len(scores.shape)
		indices = scores.argmax(axis = 1)
		prediction_Y  = np.zeros(Y.shape)
			# print prediction_Y.shape
		for i in range(0, len(indices)):
			prediction_Y[i][indices[i]] = 1	
		

	#class sklearn.svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000
	#class sklearn.svm.SVC(C=1.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)

	# 

	

	# if(len(scores.shape) == 1):
	# 	indices = (scores > 0).astype(np.int)
	# else:
	# 	indices = scores.argmax(axis = 1)
	# print indices
	# print type(clf.classes_[indices])
	# prediction_Y = clf.classes_[indices]

	# print prediction_Y

	# print indices.shape
	# print type(indices.shape)

	
	# print type(prediction_Y)
	# print prediction_Y

	# def shagun_predict()
	#     def predict(self, X):
 #        """Predict class labels for samples in X.
 #        Parameters
 #        ----------
 #        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
 #            Samples.
 #        Returns
 #        -------
 #        C : array, shape = [n_samples]
 #            Predicted class label per sample.
 #        """
 #        scores = self.decision_function(X)
 #        if len(scores.shape) == 1:
 #            indices = (scores > 0).astype(np.int)
 #        else:
 #            indices = scores.argmax(axis=1)
 #        return self.classes_[indices]


	# # prediction_Y = OneVsRestClassifier(SVC(random_state=0, verbose = 0)).fit(train_data, Y).predict(train_data)
	# # print type(prediction_Y)
	
	prediction = mlb.inverse_transform(prediction_Y)
	# # print prediction
	# for i in prediction:
	# 	print i
	# print "\n"
	# for i in train_results:
	# 	print i
	# print clf.decision_function(X)
	# # # # print Y
	# print Y
	# print prediction_Y
	evaluate.accuracy_atleast_one_match(train_results, prediction)
	evaluate.accuracy_null_results(prediction)
	evaluate.accuracy_exact_match(train_results, prediction)
	evaluate.accuracy(train_results, prediction)
	evaluate.precision(train_results, prediction)
	evaluate.recall(train_results, prediction)
	print raw_train_data.shape
	# print train_results
	# print prediction


if __name__ == "__main__":
	predict(1000, select_transform = 2, read_database = 1, one_vs_one = 0, model = "SVC")
