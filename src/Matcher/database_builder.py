# Import the load_dotenv function
from dotenv import load_dotenv
import os
import json
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


import json
import os
from langchain.docstore.document import Document

embeddings = HuggingFaceEmbeddings(model_name="dmlls/all-mpnet-base-v2-negation")

class TrialDatabaseBuilder:
    def __init__(self, json_directory, desired_fields, fields_to_concatenate):
        self.json_directory = json_directory
        self.desired_fields = desired_fields
        self.fields_to_concatenate = fields_to_concatenate
        self.docs = []
        self.ids = []

    def load_json_files(self):
        for filename in os.listdir(self.json_directory):
            if filename.endswith('.json'):
                file_path = os.path.join(self.json_directory, filename)
                with open(file_path, 'r') as file:
                    json_data = json.load(file)
                    extracted_data = {field: json_data.get(field) for field in self.desired_fields}
                    self.ids.append(extracted_data["nct_id"])
                    
                    metadata = {
                        "id": extracted_data.get("nct_id", ""),
                        "gender": extracted_data.get("gender", ""),
                        "condition": extracted_data.get("condition", ""),
                        "phase": extracted_data.get("phase", ""),
                        "minimum_age": extracted_data.get("minimum_age", ""),
                        "maximum_age": extracted_data.get("maximum_age", ""),
                        
                    }
                    metadata = {k: v for k, v in metadata.items() if v is not None}
                        
                    concatenated_string = ', '.join(str(extracted_data[field]) for field in self.fields_to_concatenate)
                    doc  = Document(page_content=concatenated_string, metadata=metadata)
                    self.docs.append(doc)
                    
    def build_vectorstore(self):
        vectorstore = Chroma.from_documents(self.docs, embeddings, persist_directory="../../data/db/", collection_name="trials")
        vectorstore.persist()
        vectorstore = None


class CriteriaDatabaseBuilder:
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
        vectorstore = Chroma.from_documents(self.docs, embeddings, persist_directory="../../data/db/", collection_name="criteria")
        vectorstore.persist()
        vectorstore = None


def main():
    load_dotenv('../.env')
    openai_access_key = os.getenv('OPENAI_ACCESS_KEY')
    huggingface_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
    json_directory = '../../data/trials_jsons/'
    desired_fields_criteria = ["nct_id", "eligibility"]

    criteriadb_builder = CriteriaDatabaseBuilder(json_directory, desired_fields_criteria)
    criteriadb_builder.load_json_files()
    criteriadb_builder.build_vectorstore()
    
    desired_fields_trials=["nct_id", "brief_title", "brief_summary", "condition", "gender", "minimum_age", "maximum_age", "phase"]
    trialdb_builder = TrialDatabaseBuilder(json_directory, desired_fields_trials)
    trialdb_builder.load_json_files()
    trialdb_builder.build_vectorstore()


if __name__ == "__main__":
    main()
    
    