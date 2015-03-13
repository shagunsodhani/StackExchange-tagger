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

test_data = "data/processed.csv"
takeword_data = "data/take_word.txt"
replaceword_data = "data/replaceword.txt"

def get_featureVector(input_size = 10000):
    #this function is meant to print the unique words with their frequency so that some potential stopwords can be removed

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

	return corpus
	vectorizer = TfidfVectorizer(min_df=1)
	a = vectorizer.fit_transform(corpus)
	U, s, V = np.linalg.svd(a.toarray(), full_matrices=True)
	


# get_featureVector(input_size = 100)
