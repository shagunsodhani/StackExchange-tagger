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
import numpy as np

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

#get_codewords()

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
					word[processed_token]+=1
	sorted_word = sorted(word.items(), key=operator.itemgetter(1), reverse = True)
	print sorted_word
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
					word[processed_token]+=1
					if processed_token not in flag:
						flag[processed_token] = 1
						idf[processed_token]+=1

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

def get_boolmatrix(input_size = 100000):
	porter_stemmer = nltk.stem.porter.PorterStemmer()
	wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()
	nltk_stopwords = nltk.corpus.stopwords.words('english')
	
	take_words = {}
	replace_words = {}

	with open(takeword_data) as infile:
		for line in infile:
			i = line.strip().split()
			for token in i:
				if token not in take_words:
					take_words[token] = 1

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
	# count
	for post in list(db.find().skip(1).limit(input_size)):
		word = {}
		body = post['body'].strip()
		for i in replace_words:
			body = body.replace(i, '')
		list_token = nltk.word_tokenize(body)
		processed_body = ""
		for token in list_token:
			processed_token = wordnet_lemmatizer.lemmatize(porter_stemmer.stem(token.strip().lower()))
			if processed_token in take_words:
				if processed_token not in word:
					processed_body+=processed_token+" "
					word[processed_token]=1
		corpus.append(processed_body.strip())

	# return corpus
	vectorizer = CountVectorizer(min_df=1)
	a = vectorizer.fit_transform(corpus)
	b = a.toarray()
	count = 0
	x, y = b.shape
	to_print = ""
	for c in np.nditer(b):
		to_print+=str(c)+"," 
		count+=1
		if(count%y == 0):
			print to_print
			to_print = ""
	# U, s, V = np.linalg.svd(a.toarray(), full_matrices=True)

get_boolmatrix()
