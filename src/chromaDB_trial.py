import chromadb
import json
client = chromadb.HttpClient()
collection = client.create_collection("NCT00001637_0", metadata={"hnsw:space": "cosine"})

json_data = json.load(open("../data/trials_jsons/NCT00001637.json"))

eligibility_criteria = json_data.get("eligibility", [])

for index, criterion in enumerate(eligibility_criteria, start=1):
    text = criterion.get("text", "")
    entities_data = criterion.get("entities_data", [])
    nct_id = json_data.get("nct_id", "")

    # Convert entities_data to JSON string
    entities_data_json = json.dumps(entities_data)

    # Create metadatas with entities_data as a JSON string
    metadatas = {"entities_data": entities_data_json}
    # Add the document to the collection
    collection.add(
        documents=[text],
        metadatas=[metadatas],  # Adjusted to have a single dictionary
        ids=["{}-{}".format(nct_id, index)]
    )
    
# print(collection.peek())


# collection = client.get_collection("clinical_trials")

# # Add docs to the collection. Can also update and delete. Row-based API coming soon!
# collection.add(
#     documents=["The patient should have a mutation status BRAF", "The patient should not have diabetes"], # we embed for you, or bring your own
#     metadatas=[{"source": "notion"}, {"source": "google-docs"}], # filter on arbitrary metadata!
#     ids=["doc1", "doc2"], # must be unique for each doc 
# )

results = collection.query(
    query_texts=["Patient's age is 25 years old.", "Patient is pregnant and lactating"],
    n_results=2,
    # where={"metadata_field": "is_equal_to_this"}, # optional filter
    # where_document={"$contains":"search_string"}  # optional filter
)  

print(results)