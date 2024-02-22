import chromadb
import os
import pandas as pd
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient()
em = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2", device="cuda")
collection = client.get_or_create_collection("eligibility_criteria_collection", metadata={"hnsw:space": "cosine"}, embedding_function=em) # cosine is the default

# Create an empty list to store filenames without extension
folder_path='../data/preprocessed_data/clinical_trials'
files = os.listdir(folder_path)
filenames_without_extension = []
for file in files:
    filename, file_extension = os.path.splitext(file)
    filenames_without_extension.append(filename)

def add_documents_collection(filenames, collection, folder_path):
    for f in filenames:
        print(f)
        df = pd.read_csv(f"{folder_path}/{f}.csv")
        df = df.dropna()
        for index, row in df.iterrows():
            text = row["sentence"]
            nct_id = row["id"]
            metadatas = {
                "criteria": row["criteria"],  # Replace "criteria" with the actual column name
                "sub-criteria": row["sub_criteria"]  # Replace "sub-criteria" with the actual column name
            }
            # Add the document to the collection
            collection.add(
                documents=[text],
                ids=["{}-{}".format(nct_id, index + 1)],
                metadatas=metadatas
            )



add_documents_collection(filenames=filenames_without_extension, collection=collection, folder_path=folder_path);
print(collection.count())
