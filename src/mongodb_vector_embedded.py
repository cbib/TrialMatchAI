import pymongo
import requests
from typing import List, Dict
from tenacity import retry, wait_random_exponential, stop_after_attempt

client = pymongo.MongoClient("mongodb+srv://abdallahmajd7:Basicmongobias72611@trialmatchai.pvx7ldb.mongodb.net/")
db = client.trialmatchai
collection = db.clinicaltrials
import openai

openai.api_key = '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'

# hf_token = "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"
# embedding_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(10))
def generate_embedding(text: str, model="text-embedding-ada-002") -> list[float]:
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]

def embed_text_fields(doc, fields):
    for field in fields:
        if field in doc and isinstance(doc[field], str):
            # Generate embeddings for the text in the specified field
            doc[f'{field}_embedding_hf'] = generate_embedding(doc[field])

def embed_eligibility_text_and_entity_text(doc):
    print(doc["nct_id"])
    # Check if the eligibility field exists and is a list
    if 'eligibility' in doc and isinstance(doc['eligibility'], list):
        for entry in doc['eligibility']:
            # Check if the entry has 'text' and 'entity_text' sub-fields
            if 'text' in entry:
                # Generate embeddings for 'text' and 'entity_text' if available
                entry['text_embedding_hf'] = generate_embedding(entry['text'])

            if 'entities_data' in entry:
                for entity_data in entry['entities_data']:
                    if 'entity_text' in entity_data:
                        # Generate embeddings for 'entity_text' if available
                        entity_data['entity_text_embedding_hf'] = generate_embedding(entity_data['entity_text'])

# List of fields to embed
fields_to_embed = ['eligibility', 'brief_title']

# Embed specified fields for each document in the collection
for doc in collection.find({}):
    embed_text_fields(doc, fields_to_embed)

    # Special case for 'eligibility' field
    embed_eligibility_text_and_entity_text(doc)

    # Update the document in the collection
    collection.replace_one({'_id': doc['_id']}, doc)


query = """The patient has mutational status of EGFR and ALK. 
The patient suffers from stage 4 lung cancer. The patient has not received any prior treatment. 
The patient has a life expectancy of at least 3 months. 
The patient has an ECOG performance status of 0 or 1. 
The patient has measurable disease. 
The patient has adequate organ function. 
The patient has a negative pregnancy test.
The patient has a history of interstitial lung disease. 
The patient has a history of non-infectious pneumonitis. 
The patient has no history of radiation pneumonitis. 
The patient has no history of drug-induced pneumonitis. 
The patient has no history of idiopathic pulmonary fibrosis.
The patient has no history of organizing pneumonia. 
The patient has no history of autoimmune disease. 
The patient has no history of systemic lupus erythematosus. 
The patient has no history of sarcoidosis. 
The patient has no history of vasculitis. 
The patient has no history of hypophysitis. 
The patient has no history of uveitis. 
The patient has no history of iritis.
The patient has a history of hepatitis B. 
The patient has no history of hepatitis C. 
The patient has no history of human immunodeficiency virus. 
The patient has no history of tuberculosis. 
The patient has no history of active infection. 
The patient has a history of severe infection. 
The patient has no history of severe or uncontrolled cardiovascular disease. 
The patient has no history of severe or uncontrolled hypertension.
The patient has no history of severe or uncontrolled diabetes.
The patient has no history of severe or uncontrolled hyperlipidemia. 
The patient has no history of severe or uncontrolled hypertriglyceridemia.
The patient has no history of severe or uncontrolled hypercholesterolemia. 
The patient has a history of severe or uncontrolled hypomagnesemia. 
The patient has a history of severe or uncontrolled hypophosphatemia. 
The patient has no history of severe or uncontrolled hypovitaminosis D. 
The patient has a history of severe or uncontrolled hyperthyroidism""" 

results = collection.aggregate([
  {"$vectorSearch": {
    "queryVector": generate_embedding(query),
    "path": "plot_embedding_hf",
    "numCandidates": 100,
    "limit": 4,
    "index": "PlotSemanticSearch",
      }}
])

