#!/bin/bash

apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 7F0CEB10
echo "deb http://repo.mongodb.org/apt/ubuntu "$(lsb_release -sc)"/mongodb-org/3.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-3.0.list
apt-get update
pip install pymongo numpy nltk BeautifulSoup cython sparsesvd
apt-get install -y mongodb-org python-scipy scikit-learn
service mongod start
python -m nltk.downloader
#To download nltk datasets
