import chromadb
client = chromadb.HttpClient()
collection = client.get_collection("clinical_trials")

# Add docs to the collection. Can also update and delete. Row-based API coming soon!
collection.add(
    documents=["The patient should have a mutation status BRAF", "The patient should not have diabetes"], # we embed for you, or bring your own
    metadatas=[{"source": "notion"}, {"source": "google-docs"}], # filter on arbitrary metadata!
    ids=["doc1", "doc2"], # must be unique for each doc 
)

results = collection.query(
    query_texts=["Patient with BRAF mutation", "Patient without no diabetes"],
    n_results=2,
    # where={"metadata_field": "is_equal_to_this"}, # optional filter
    # where_document={"$contains":"search_string"}  # optional filter
)  

print(results)