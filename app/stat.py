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

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time
import numpy as np
import pickle

stopword_data = "data/stopword.txt"
replaceword_data = "data/replaceword.txt"
test_data = "data/processed.csv"
takeword_data = "data/take_word.txt"

def get_codewords():
	#this function is meant prints all the code segments
	db = mongo.connect()
	code_word = {}
	for post in db.find():
		code_temp = post['code'].split()
		for i in code_temp:
			if i not in code_word:
				try:
					print i
				except UnicodeEncodeError as e:
					pass
				code_word[i] = 1
		print "\n"

def get_bodywords():
    #this function is meant to print the unique words with their frequency so that some potential stopwords can be removed
	porter_stemmer = nltk.stem.porter.PorterStemmer()
	wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()
	nltk_stopwords = nltk.corpus.stopwords.words('english')
	stopwords = {}
	replace_words = {}
	stopword_count = 0
	takenword_count = 0

	with open(stopword_data) as infile:
		for line in infile:
			i = line.strip().split()
			for token in i:
				a = wordnet_lemmatizer.lemmatize(porter_stemmer.stem(token))
				if a not in stopwords:
					stopwords[a] = 1

	for token in nltk_stopwords:
		a = wordnet_lemmatizer.lemmatize(porter_stemmer.stem(token))
		if a not in stopwords:
			stopwords[a] = 1

	for a in string.punctuation:
		if a not in replace_words:
			replace_words[a] = 1
	
	with open(replaceword_data) as infile:
		for line in infile:
			a = line.strip()
			if a not in replace_words:
				replace_words[a] = 1			

	db= mongo.connect()
	word = {}

	for post in db.find():
		body = post['body'].strip()
		for i in replace_words:
			body = body.replace(i, '')
		list_token = nltk.word_tokenize(body)
		for token in list_token:
			# print token
			processed_token = wordnet_lemmatizer.lemmatize(porter_stemmer.stem(token.strip().lower()))
			if processed_token not in stopwords:
				if processed_token not in word:
					word[processed_token]=1
					# print processed_token
				else:
					word[processed_token]=1
	sorted_word = sorted(word.items(), key=operator.itemgetter(1), reverse = True)
	#print sorted_word
	for i in sorted_word:
		try:
			print i[0], " : ",i[1] 
		except UnicodeEncodeError as e:
			print "Unicode Error : ", i[1]

def get_idf():
    #this function is meant to print the unique words with their frequency so that some potential stopwords can be removed

	porter_stemmer = nltk.stem.porter.PorterStemmer()
	wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()
	nltk_stopwords = nltk.corpus.stopwords.words('english')
	stopwords = {}
	replace_words = {}
	stopword_count = 0
	takenword_count = 0

	with open(stopword_data) as infile:
		for line in infile:
			i = line.strip().split()
			for token in i:
				a = wordnet_lemmatizer.lemmatize(porter_stemmer.stem(token))
				if a not in stopwords:
					stopwords[a] = 1

	for token in nltk_stopwords:
		a = wordnet_lemmatizer.lemmatize(porter_stemmer.stem(token))
		if a not in stopwords:
			stopwords[a] = 1

	for a in string.punctuation:
		if a not in replace_words:
			replace_words[a] = 1
	
	with open(replaceword_data) as infile:

		for line in infile:
			a = line.strip()
			if a not in replace_words:
				replace_words[a] = 1			

	db= mongo.connect()
	word = {}
	idf = {}
	flag = {} 

	for post in db.find():
		body = post['body'].strip()
		flag = {}
		for i in replace_words:
			body = body.replace(i, '')
		list_token = nltk.word_tokenize(body)
		for token in list_token:
			# print token
			processed_token = wordnet_lemmatizer.lemmatize(porter_stemmer.stem(token.strip().lower()))
			if processed_token not in stopwords and not (processed_token.isdigit()):
				if processed_token not in word:
					word[processed_token]=1
					idf[processed_token] = 1
					flag[processed_token] = 1
					# print processed_token
				else:
					word[processed_token]=1
					if processed_token not in flag:
						flag[processed_token] = 1
						idf[processed_token]=1

	for i in idf:
		if idf[i] > 7:
			try:
				print i
			except UnicodeEncodeError as e:
				pass

	# sorted_idf = sorted(idf.items(), key=operator.itemgetter(1), reverse = True)
	# for i in sorted_idf:
	# 	try:
	# 		print i[0], " : ",i[1] 
	# 	except UnicodeEncodeError as e:
	# 		print "Unicode Error : ", i[1]

def get_trainingdata(input_size = 100000, select_transform = 1, read_database = 1, to_print = 0, mode = "multiclass", repeat = 0, max_number_of_tags = 5):
	'''
		generate training data
		if read_database == 0:
			All other options are ignored
		if mode == multilabel
			repeat option is ignored

	'''

	fname_feature = "trainfeaturematrix.csv"
	fname_result = "trainresultmatrix.csv"
	fname_result_pickle = "trainresultmatrix"

	if read_database == 0:
		t0 = time()
		a = np.loadtxt(fname_feature, delimiter = ",")
		print("Loaded feature matrix for training from File in %fs" % (time() - t0))
		# print "input_size = ", input_size
		# print a.size
		trainingdata_features = a.reshape(input_size, a.size/input_size)

		t0 = time()
		print("Loaded result matrix for training from File in %fs" % (time() - t0))
		#print "input_size = ", input_size
		#print a.size
		with open(fname_python, 'rb') as f:
			    trainingdata_result = pickle.load(f)
		#train = a.reshape(input_size, a.size/input_size)

		return trainingdata_features, trainingdata_result

	else:

		porter_stemmer = nltk.stem.porter.PorterStemmer()
		wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()
		nltk_stopwords = nltk.corpus.stopwords.words('english')
		
		take_words = {}
		replace_words = {}
		stopwords = {}

		with open(stopword_data) as infile:
			for line in infile:
				i = line.strip().split()
				for token in i:
					a = wordnet_lemmatizer.lemmatize(porter_stemmer.stem(token))
					if a not in stopwords:
						stopwords[a] = 1

		for a in string.punctuation:
			if a not in replace_words:
				replace_words[a] = 1
		
		with open(replaceword_data) as infile:
			for line in infile:
				a = line.strip()
				if a not in replace_words:
					replace_words[a] = 1			

		db = mongo.connect()
		corpus = []
		tag_set = set()
		question_tag = {}
		question_count = 0
		# counter = 0
		trainingdata_result = []

		if mode == "multilabel":

			for post in list(db.find().skip(1).limit(input_size*(max_number_of_tags+1 ))):

				#not fool proof
				if(len(post['tag']) <= max_number_of_tags):
					question_tag[question_count] = []

					trainingdata_result.append(post['tag'])

					body = post['body'].strip()
					for i in replace_words:
						body = body.replace(i, '')
					list_token = nltk.word_tokenize(body)
					processed_body = ""
					for token in list_token:
						processed_token = wordnet_lemmatizer.lemmatize(porter_stemmer.stem(token.strip().lower()))
						if processed_token not in stopwords and not (processed_token.isdigit()):
							processed_body+=processed_token+" "
					corpus.append(processed_body.strip())

					for i in post['tag']:
						question_tag[question_count].append(i)
						tag_set.add(i)
					question_count+=1

				if(question_count>=input_size):
					break


			#entire point of writing to csv is to use the data with matlab
			sorted_taglist = sorted(tag_set)
			tag_dict = {}
			tag_count = 0
			# print "size of set"
			# print len(tag_set)
			for i in sorted_taglist:
				# print i
				tag_dict[i] = tag_count
				tag_count+=1
			# print "number of unique tags = "+str(tag_count)
			train_matrix = np.zeros((input_size, tag_count), dtype = np.int)
			for i in question_tag:
				for j in question_tag[i]:
					train_matrix[i][tag_dict[j]]=1
			with open(fname_result_pickle, 'wb') as f:
				pickle.dump(trainingdata_result, f)

			if(select_transform == 1):
				transform = CountVectorizer(min_df=1)
			elif(select_transform == 2):
				transform = TfidfVectorizer(min_df=1)
			a = transform.fit_transform(corpus)
			# print transform.get_feature_names()
			trainingdata_features = a.toarray()
			# print trainingdata_features
			np.savetxt(fname_feature, trainingdata_features, delimiter=",")

		elif mode == "multiclass":
			if repeat == 0:
				for post in list(db.find().skip(1).limit(input_size*(max_number_of_tags+1))):
					#not fool proof

					if(len(post['tag']) <= max_number_of_tags):
						# print len(post['tag'])
						body = post['body'].strip()
						for i in replace_words:
							body = body.replace(i, '')
						list_token = nltk.word_tokenize(body)
						processed_body = ""
						for token in list_token:
							processed_token = wordnet_lemmatizer.lemmatize(porter_stemmer.stem(token.strip().lower()))
							if processed_token not in stopwords and not (processed_token.isdigit()):
								processed_body+=processed_token+" "
						
						for i in post['tag']:
							# print i
							question_tag[question_count] = []
							question_tag[question_count].append(i)
							#can be done in a single step - then do nit
							tag_set.add(i)
							question_count+=1
							corpus.append(processed_body.strip())
							trainingdata_result.append([i])
							if(question_count >= input_size):
								break

					if(question_count >= input_size):
						break

				# print corpus
				# print 
				#entire point of writing to csv is to use the data with matlab
				sorted_taglist = sorted(tag_set)
				tag_dict = {}
				tag_count = 0
				# print "size of set"
				# print len(tag_set)
				for i in sorted_taglist:
					# print i
					tag_dict[i] = tag_count
					tag_count+=1
				# print "number of unique tags = "+str(tag_count)
				train_matrix = np.zeros((input_size, tag_count), dtype = np.int)
				for i in question_tag:
					for j in question_tag[i]:
						train_matrix[i][tag_dict[j]]=1
				with open(fname_result_pickle, 'wb') as f:
					pickle.dump(trainingdata_result, f)

				if(select_transform == 1):
					transform = CountVectorizer(min_df=1)
				elif(select_transform == 2):
					transform = TfidfVectorizer(min_df=1)
				a = transform.fit_transform(corpus)
				# print transform.get_feature_names()
				trainingdata_features = a.toarray()
				# print trainingdata_features
				np.savetxt(fname_feature, trainingdata_features, delimiter=",")


		if to_print == 1:
			for i in trainingdata_features:
				to_print = ""
				for j in i:
					to_print+=str(j)+", "
					to_print = to_print[:-2]
				print to_print
	
		return trainingdata_features, trainingdata_result

if __name__ == "__main__":
#	get_trainmatrix(input_size = 10000)
	get_featurematrix(200, select_transform = 2, read_database = 1)
	get_trainmatrix(200, read_database = 1)

