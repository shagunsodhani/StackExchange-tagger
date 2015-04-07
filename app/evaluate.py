def accuracy_atleast_one_match(actual, prediction, verbose = 1):
	'''
		actual - list of actual results
		prediction - list of predicted results
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
		actual - list of actual results
		prediction - list of predicted results
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
		actual - list of actual results
		prediction - list of predicted results
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


def hamming_loss(actual, prediction, verbose = 1):
	'''
		actual - list of actual results
		prediction - list of predicted results
		defination taken from http://lpis.csd.auth.gr/publications/tsoumakas-ijdwm.pdf
	'''

	length = len(actual)
	hamming_loss = 0.0
	for i in range(0, length):
		yi = set()
		zi = set()

		for j in actual[i]:
			yi.add(j)

		for j in prediction[i]:
			zi.add(j)

		hamming_loss+=(len(yi.symmetric_difference(zi))+0.0)/len(zi)

	hamming_loss = hamming_loss/length
	if (verbose):
		print "Hamming Loss = "+str(hamming_loss)
	return hamming_loss


def accuracy(actual, prediction, verbose = 1):
	'''
		actual - list of actual results
		prediction - list of predicted results
		defination taken from http://lpis.csd.auth.gr/publications/tsoumakas-ijdwm.pdf
	'''

	length = len(actual)
	accuracy = 0.0
	for i in range(0, length):
		yi = set()
		zi = set()

		for j in actual[i]:
			yi.add(j)

		for j in prediction[i]:
			zi.add(j)

		accuracy+=(len(yi.intersection(zi))+0.0)/len(yi.union(zi))

	accuracy = accuracy/length
	if (verbose):
		print "Accuracy (Godbole & Sarawagi) = "+str(accuracy)
	return accuracy


def precision(actual, prediction, verbose = 1):
	'''
		actual - list of actual results
		prediction - list of predicted results
		defination taken from http://lpis.csd.auth.gr/publications/tsoumakas-ijdwm.pdf
	'''

	length = len(actual)
	precision = 0.0
	for i in range(0, length):
		yi = set()
		zi = set()

		for j in actual[i]:
			yi.add(j)

		for j in prediction[i]:
			zi.add(j)

		precision+=(len(yi.intersection(zi))+0.0)/len(zi)
		# print zi
	precision = precision/length
	if (verbose):
		print "Precision (Godbole & Sarawagi) = "+str(precision)
	return precision

def recall(actual, prediction, verbose = 1):
	'''
		actual - list of actual results
		prediction - list of predicted results
		defination taken from http://lpis.csd.auth.gr/publications/tsoumakas-ijdwm.pdf
	'''

	length = len(actual)
	recall = 0.0
	for i in range(0, length):
		yi = set()
		zi = set()

		for j in actual[i]:
			yi.add(j)

		for j in prediction[i]:
			zi.add(j)

		recall+=(len(yi.intersection(zi))+0.0)/len(yi)

	recall = recall/length
	if (verbose):
		print "Recall (Godbole & Sarawagi) = "+str(recall)
	return recall