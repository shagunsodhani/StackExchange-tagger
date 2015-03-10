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
				for i in replace_words:
					body = body.replace(i, ' ')
				list_token = nltk.word_tokenize(body)
				for token in list_token:
					processed_token = wordnet_lemmatizer.lemmatize(porter_stemmer.stem(token.strip().lower()))
					if(processed_token in stopwords):
						stopword_count+=1
					else:
						takenword_count+=1
						try:
							print processed_token
						except UnicodeEncodeError as e:
							print "Unicode Encode Error ", e
			print "\n"
	print "stopword_count : ", stopword_count
	print "takenword_count : ", takenword_count	

def fetch_top_tags(k = 100):
	#script to fetch top k most popular tags from raw data
	tags = {}
	with open(test_data) as infile:
		for line in infile:
			striped_line = line.strip().replace('"','')
			if striped_line :
				a = striped_line.split(',',2)
				post_id = str(a[0])
				title = str(a[1])
				a = a[2].rsplit(',',1)
				tag_list = a[1].split(' ')
				for tag in tag_list:
					if tag not in tags:
						tags[tag]=1
					else:
						tags[tag]+=1
	sorted_tags = sorted(tags.items(), key=operator.itemgetter(1), reverse = True)
	for i in range(0, k):
		print sorted_tags[i][0]	 

#preprocess_dataset()
#remove_stopwords()
fetch_top_tags()
