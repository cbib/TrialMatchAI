# Import the load_dotenv function
from dotenv import load_dotenv
import os
import json
from langchain.docstore.document import Document
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import Chroma


class DatabaseBuilder:
    def __init__(self, json_directory, desired_fields):
        self.json_directory = json_directory
        self.desired_fields = desired_fields
        self.docs = []

    def load_json_files(self):
        for filename in os.listdir(self.json_directory):
            if filename.endswith('.json'):
                file_path = os.path.join(self.json_directory, filename)
                with open(file_path, 'r') as file:
                    json_data = json.load(file)
                    extracted_data = {field: json_data.get(field) for field in self.desired_fields}
                    eligibility_criteria = json_data.get("eligibility")
                    if eligibility_criteria is not None:
                        for index, criterion in enumerate(eligibility_criteria):
                            metadata = {
                                "nct_id": extracted_data['nct_id'],
                                "idx": index + 1,
                            }
                            metadata["criteria_type"] = criterion["entities_data"][0]["field"]
                            for i, entity in enumerate(criterion["entities_data"]):
                                for key, value in entity.items():
                                    if key != "field":
                                        metadata[f"{key}_{i + 1}"] = value
                            doc = Document(page_content=criterion["text"], metadata=metadata)
                            self.docs.append(doc)

    def build_vectorstore(self):
        vectorstore = Chroma.from_documents(self.docs, SentenceTransformerEmbeddings(), persist_directory="../../data/db/", collection_name="criteria")
        vectorstore.persist()
        vectorstore = None


def main():
    load_dotenv('../.env')
    openai_access_key = os.getenv('OPENAI_ACCESS_KEY')
    huggingface_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
    json_directory = '../../data/trials_jsons/'
    desired_fields = ["nct_id", "eligibility"]

    builder = DatabaseBuilder(json_directory, desired_fields)
    builder.load_json_files()
    builder.build_vectorstore()


if __name__ == "__main__":
    main()
    
    