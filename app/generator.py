import sys
import os
import json
import time
import string

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
tag_data = "data/tag.txt"

def generate_taglist(k):
	#genetrates a dict of top k tags"
	tags = {}
	with open(tag_data) as infile:
		for line in infile:
			tag = line.strip()
			if tag not in tags:
				tags[tag] = 0
	return tags

def generate_data(tag_count, question_count):
	#generate data using top 'tag_count' number of tags and 'question_count' number of questions
    db = mongo.connect()
    tags = generate_taglist(tag_count)
    count = 0
    with open(test_data) as infile:
        for line in infile:
            striped_line = line.strip()
            if striped_line:
                a = striped_line.split(',', 2)
                post_id = str(a[0]).replace('\"', '').strip()
                title = str(a[1]).replace('\"', '').strip()
                a = a[2].rsplit(',', 1)
                tag_list = a[1].split()
                for tag in tag_list:
                    if tag not in tags:
                        continue
                count+=1;
                body = a[0]
                code = ""
                soup = BeautifulSoup(body)
                body = soup.get_text()
                for code_snippet in soup.find_all('code'):
                    temp_code = code_snippet.get_text().strip()
                    code+= temp_code + "\n"
                    body = body.replace(temp_code, "")
                body = ' '.join(body.split())
                post = {}
                post['post_id'] = post_id
                post['title'] = title
                post['body'] = body
                post['tag'] = tag_list
                post['code'] = code
                mongo_id = db.insert(post)
                if(count%10000 == 0):
                    print count, " number of questions processed"
                if(count > question_count):
                    break;

generate_data(100, 100000)
