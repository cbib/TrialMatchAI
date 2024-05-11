import socket
import struct
import json
import argparse
import torch
from transformers import AutoTokenizer, pipeline
import warnings

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

def get_dictionaries_of_specific_entities(list_of_dicts, key, values):
    return [d for d in list_of_dicts if any(val in d.get(key, []) for val in values)]

class BioMedNERMacrobbat:
    def __init__(self, params):
        self.params = params
        self.entities = ["Sign_symptom", "Biological_structure", "Date", "Duration", "Time", "Frequency", 
                         "Severity", "Lab_value", "Dosage", "Diagnostic_procedure", "Therapeutic_procedure", "Medication", 
                         "Clinical_event", "Outcome", "History", "Subject", "Family_history", "Detailed_description", "Area"]
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.params.model_path_or_name, model_max_length=512, truncation=True)
        self.ner_pipeline = pipeline("ner", model=self.params.model_path_or_name, tokenizer=self.tokenizer, aggregation_strategy="first", device=self.params.device)

    def recognize(self, text):
        result_entities = self.ner_pipeline(text)
        filtered_entities = get_dictionaries_of_specific_entities(result_entities, "entity_group", self.entities)
        # Ensure all data is JSON serializable
        clean_entities = [{
            "entity_group": d["entity_group"],
            "score": float(d["score"]),  # Convert numpy float32 to Python float
            "text": d["word"],
            "start": int(d["start"]),  # Ensure start and end are ints
            "end": int(d["end"])
        } for d in filtered_entities]
        return clean_entities


def handle_client_connection(connection, model):
    while True:
        # Receive data from the client
        data_length = connection.recv(2)
        if not data_length:
            break
        data_length = struct.unpack('>H', data_length)[0]
        text = connection.recv(data_length).decode('utf-8')

        # Process the text using the model
        entities = model.recognize(text)

        # Send back the results
        response = json.dumps(entities)
        connection.send(struct.pack('>H', len(response)) + response.encode('utf-8'))

    connection.close()

def run_server(model, args):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((args.host, args.port))
        server_socket.listen(5)
        print(f"Server listening on {args.host}:{args.port}")
        while True:
            conn, _ = server_socket.accept()
            handle_client_connection(conn, model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BioMedNERMacrobbat Server")
    parser.add_argument("--model_path_or_name", type=str, help="Path to the model")
    parser.add_argument("--device", type=str, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"), help="Device to use")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    args = parser.parse_args()
    
    ner_model = BioMedNERMacrobbat(args)
    run_server(ner_model, args)
