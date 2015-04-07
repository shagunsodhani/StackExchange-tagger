def accuracy_atleast_one_match(actual, prediction, verbose = 1):
	'''
		actual - list of actual null_results
		prediction - list of predicted null_results
	'''

	length = len(actual)
	count = 0.0
	for i in range(0, length):
		flag = 0
		for j in prediction[i]:
			if j in actual[i]:
				flag = 1
		count+=flag
		# print "Result : "+str(result[i])
		# print "Prediction : "+str(prediction[i])

	if(verbose): 
		print "Accuracy for matching atleast one = "+str(count/length)
	return count/length

def accuracy_null_results(prediction, verbose = 1):
	''' 
		actual - list of actual null_results
		prediction - list of predicted null_results
	'''
	length = len(prediction)
	count = 0.0
	for i in range(0, length):
		if not prediction[i]:
			count+=1
	if(verbose):
		print "Percentage of null_results = "+str(count/length)
	return count/length

def accuracy_exact_match(actual, prediction, verbose = 1):
	'''
		actual - list of actual null_results
		prediction - list of predicted null_results
	'''

	length = len(actual)
	count = 0.0
	for i in range(0, length):
		flag = 1
		if len(prediction[i]) == len(actual[i]):
			for j in prediction[i]:
				if j not in actual[i]:
					flag = 0
		else:
			flag = 0
		count+=flag
		# print "Result : "+str(result[i])
		# print "Prediction : "+str(prediction[i])

	if(verbose): 
		print "Accuracy for exact matching = "+str(count/length)
	return count/length