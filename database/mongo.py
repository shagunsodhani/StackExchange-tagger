#! /usr/bin/python
#---------------------------------------------------------Import Modules----------------------------------------------------------------------#

import os
from ConfigParser import ConfigParser

try:
    import pymongo
except ImportError as exc:
    print("Error: failed to import settings module ({})".format(exc))

def connect(app_name = "tagger", config_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../config', 'config.cfg') ):

    '''Open connection to mongodb and return db object to perform queries'''
    config=ConfigParser()
    config.read(config_path)
    host=config.get(app_name,"host")
    port=config.get(app_name,"port")
    db_name=config.get(app_name,"db_name")
    collection_name=config.get(app_name, "collection_name")
    try:
        client = pymongo.MongoClient(host, int(port))
        db = client[db_name]
        return db['store']
    except pymongo.errors, e:
        print "ERROR %d IN CONNECTION: %s" % (e.args[0], e.args[1])
        return 0
