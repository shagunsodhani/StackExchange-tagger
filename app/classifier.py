import sys
import os
import json
import string
import operator
import pickle
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

def predict(input_size = 100000, select_transform = 1, read_database = 1, one_vs_one = 0, model = "LinearSVC", mode = "multilable", repeat = 0, k = 0.8, max_number_of_tags = 5, max_iter = 100000, use_cache = 0):

	to_print = 0
	raw_train_data, raw_train_results = stat.get_trainingdata(input_size, select_transform = select_transform, read_database = read_database, to_print = to_print, mode = mode, repeat = repeat, max_number_of_tags = max_number_of_tags)
	t0 = time()
	# k = 0.8

	# # print raw_train_data
	# print raw_train_data
	# print raw_train_results

	split_point = int(k*input_size)
	# print split_point
	train_data = raw_train_data[0:split_point,:]
	train_results = raw_train_results[0:split_point]
	# print train_results
	# print train_data
	# print train_results

	test_data = raw_train_data[split_point:,:]
	test_results = raw_train_results[split_point:]
	# print test_results
	
	fname_U = "SVD_U.txt"
	fname_V = "SVD_V.txt"
	fname_S = "SVD_S.txt"

	if use_cache==1:
		with open(fname_U, 'rb') as f:
				U = pickle.load(f)
		with open(fname_V, 'rb') as f:
				V = pickle.load(f)
		with open(fname_S, 'rb') as f:
				s = pickle.load(f)
		print "Using SVD from file"
	else:
		U, s, V = np.linalg.svd(train_data, full_matrices=True)
		with open(fname_U, 'wb') as f:
				pickle.dump(U, f)
		with open(fname_V, 'wb') as f:
				pickle.dump(V, f)
		with open(fname_S, 'wb') as f:
				pickle.dump(s, f)
		print "Using SVD by calculation"

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
	train_X = np.dot(train_data, processedV)
	test_X = np.dot(test_data, processedV)
	
	# X = X_raw[0:k*input_size + 1, :]
	# test_X = X_raw[k*input_size+1:,:]

	
	# print "count = "+str(count)
	# print "V.shape = "+str(V.shape)
	# print "s.shape = "+str(s.shape)
	# x = np.delete(V, np.s_[count::1], 0)
	# print "x.shape = "+str(x.shape)
	# print "raw_train_data.shape = "+str(raw_train_data)
	# print "processedV.shape = "+str(processedV.shape)

	#can use splicing instead of delete
	
	# print "X.shape = "+str(X.shape)	

	# train_results = stat.get_trainmatrix(input_size, read_database = read_database, to_print = to_print)
	
	mlb = MultiLabelBinarizer()
	trainingdata_results = mlb.fit_transform(raw_train_results)
	# print train_results
	train_Y = trainingdata_results[0:split_point,:]
	test_Y = trainingdata_results[split_point+1:,:]

	# print train_Y
	# test_Y = mlb.fit_transform(test_results)
	# print test_results


	# print Y.shape
	# test_X = X[0:k*input_size,:]
	# print train_X
	# print train_Y
	# print train_results

	if(one_vs_one == 1):
		clf = OneVsOneClassifier(svm.LinearSVC(random_state=0, max_iter =10000, verbose = 0))
		prediction_Y  = clf.fit(X, Y).predict(X)
	else:
		if model == "LinearSVC":
			print "Showing Results for one vs rest multilabel classifier using LinearSVC model"
			clf = OneVsRestClassifier(svm.LinearSVC(random_state=0, dual = True, max_iter = max_iter, verbose = 0, C = 0.001, loss = "squared_hinge", multi_class="crammer_singer"))
			
		elif model == "SVC":
			print "Showing Results for one vs rest multilabel classifier using SVC model"
			clf = OneVsRestClassifier(svm.SVC(C = 0.001, kernel = 'poly',  max_iter = max_iter, verbose = 0, degree = 3))
		clf.fit(train_X, train_Y)
		print clf.get_params
		scores = clf.decision_function(test_X)
		scores_train = clf.decision_function(train_X)

		indices = scores.argmax(axis = 1)
		indices_train = scores_train.argmax(axis = 1)

		prediction_Y  = np.zeros(scores.shape)
		prediction_train = np.zeros(scores_train.shape)

			# print prediction_Y.shape
		for i in range(0, len(indices)):
			prediction_Y[i][indices[i]] = 1

		for i in range(0, len(indices_train)):
			prediction_train[i][indices_train[i]] = 1



	#class sklearn.svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000
	#class sklearn.svm.SVC(C=1.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)

	prediction = mlb.inverse_transform(prediction_Y)
	# print prediction
	# for i in prediction:
	# 	print i
	# print "\n"
	# for i in test_results:
	# 	print i
	# print clf.decision_function(test_X)
	# # # # print Y
	# print test_Y
	print "Testing Error : "
	evaluate.accuracy_atleast_one_match(test_results, prediction)
	evaluate.accuracy_null_results(prediction)
	evaluate.accuracy_exact_match(test_results, prediction)
	evaluate.accuracy_multilabel(test_results, prediction)
	evaluate.precision_multilabel(test_results, prediction)
	evaluate.recall_multilabel(test_results, prediction)
	evaluate.hamming_loss_multilabel(test_results, prediction)
	
	# print train_results
	# print prediction
	# print prediction
	print "Training Error : "
	prediction = mlb.inverse_transform(prediction_train)
	# for i in prediction:
	# 	print i
	# print "\n"
	# for i in test_results:
	# 	print i
	# print clf.decision_function(test_X)
	# # # # print Y
	# print test_Y
	# print prediction_Y
	evaluate.accuracy_atleast_one_match(train_results, prediction)
	evaluate.accuracy_null_results(prediction)
	evaluate.accuracy_exact_match(train_results, prediction)
	evaluate.accuracy_multilabel(train_results, prediction)
	evaluate.precision_multilabel(train_results, prediction)
	evaluate.recall_multilabel(train_results, prediction)
	evaluate.hamming_loss_multilabel(train_results, prediction)
	
	# print raw_train_data.shape




if __name__ == "__main__":
	predict(10000, select_transform = 2, read_database = 1, one_vs_one = 0, model = "LinearSVC", mode="multiclass", repeat = 0, k = 0.8, max_number_of_tags = 2, max_iter = 10000, use_cache = 1)
