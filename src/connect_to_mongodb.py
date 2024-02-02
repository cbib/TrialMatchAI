# Connect to MongoDB server
import pymongo
import os
def connect_mongodb():
    # MongoDB connection string
    mongodb_link = os.environ.get('MONGODB_LINK')
    client = pymongo.MongoClient(mongodb_link)
    db = client["AIRegulation"]
    collection = db["mifid2"]
    return collection

connect_mongodb()