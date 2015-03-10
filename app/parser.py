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

dataset = str(sys.argv[1])

def remove_blanks():
	count = 0
	with open(dataset) as infile:
		for line in infile:
			if(line[-3:]=="\"\r\n"):
				#End of one post
				print line
			else:
				print line.strip(),
			count+=1
	print count

remove_blanks()
