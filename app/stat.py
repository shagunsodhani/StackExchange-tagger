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

test_data = "data/processed.csv"
stopword_data = "data/stopword.txt"
replaceword_data = "data/replaceword.txt"

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
			body = body.replace(i, ' ')
		list_token = nltk.word_tokenize(body)
		for token in list_token:
			processed_token = wordnet_lemmatizer.lemmatize(porter_stemmer.stem(token.strip().lower()))
			if processed_token not in stopwords:
				if processed_token not in word:
					word[i]=1
				else:
					word[i]+=1
	sorted_word = sorted(word.items(), key=operator.itemgetter(1), reverse = True)
	for i in sorted_word:
		try:
			print i[0], " : ",i[1] 
		except UnicodeEncodeError as e:
			print "Unicode Error : ", i[1]

get_bodywords()
