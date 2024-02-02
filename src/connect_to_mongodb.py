# Connect to MongoDB server
import pymongo
def connect_mongodb():
    # MongoDB connection string
    mongo_url = "mongodb+srv://abdallahmajd7:Basicmongobias72611@trialmatchai.pvx7ldb.mongodb.net/"
    client = pymongo.MongoClient(mongo_url)
    db = client["AIRegulation"]
    collection = db["mifid2"]
    return collection

connect_mongodb()