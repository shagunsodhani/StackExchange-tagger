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

try:
	from bs4 import BeautifulSoup
except ImportError as exc:
	print("Error: failed to import settings module ({})".format(exc))

test_data = "data/processed.csv"
stopword_data = "data/stopword.txt"

def preprocess_dataset():
	count = 0
	with open(test_data) as infile:
		for line in infile:
			if(line[-3:]=="\"\r\n"):
				#End of one post
				print line.strip()
			else:
				print line.strip(),
			count+=1
	# print count

def remove_stopwords():

	porter_stemmer = nltk.stem.porter.PorterStemmer()
	wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()
	nltk_stopwords = nltk.corpus.stopwords.words('english')
	
	stopwords = {}
	with open(stopword_data) as infile:
		for line in infile:
			a = line.strip()
			if a not in stopwords:
				stopwords[a] = 1

	with open(test_data) as infile:
		for line in infile:
			striped_line = line.strip()
			if striped_line :
				a = striped_line.split(',',2)
				post_id = str(a[0])
				title = str(a[1])
				a = a[2].rsplit(',',1)
				tag_list_string = a[1]
				body = a[0]
				#print body  
				soup = BeautifulSoup(body)
				body = soup.get_text()
				print body
			#if striped_line:
				#print striped_line.split(',',4)
			#if striped_line:
			#	body = striped_line.split(",", 2)[2]
			#	list_token = nltk.word_tokenize(body)
			#	for token in list_token:
			#		processed_token = wordnet_lemmatizer.lemmatize(porter_stemmer.stem(token))
			#		print processed_token
				
#preprocess_dataset()
remove_stopwords()
