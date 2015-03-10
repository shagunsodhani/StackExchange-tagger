import sys
import os
import json
import time

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

test_data = "data/processed.csv"
stopword_data = "data/stopword.txt"

def remove_blanks():
	count = 0
	with open(test_data) as infile:
		for line in infile:
			if(line[-3:]=="\"\r\n"):
				#End of one post
				print line
			else:
				print line.strip(),
			count+=1
	# print count

def remove_stopwords():

	porter_stemmer = nltk.stem.porter.PorterStemmer()
	wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()

	stopwords = {}
	with open(stopword_data) as infile:
		for line in infile:
			a = line.strip()
			if a not in stopwords:
				stopwords[a] = 1

	with open(test_data) as infile:
		for line in infile:
			striped_line = line.strip()
			if striped_line:
				body = striped_line.split(",", 2)[2]
				list_token = nltk.word_tokenize(body)
				for token in list_token:
					processed_token = wordnet_lemmatizer.lemmatize(porter_stemmer.stem(token))
					print processed
				
# remove_blanks()
remove_stopwords()