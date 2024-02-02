import os
import pandas as pd
import json
import pymongo

client = pymongo.MongoClient("mongodb+srv://abdallahmajd7:Basicmongobias72611@trialmatchai.pvx7ldb.mongodb.net/")
db = client["trialmatchai"]
collection = db["clinicaltrials"]

def update_json_from_dataframe_and_add_to_mongoDB(jsons_directory, dataframes_directory):
    # Create a list to store the filenames in the dataframe repository
    dataframe_filenames = []

    # Iterate through each file in the dataframe directory and add the filename (without the extension) to the list
    for dataframe_file in os.listdir(dataframes_directory):
        if dataframe_file.endswith('.csv'):
            dataframe_filenames.append(dataframe_file.split('.')[0])

    # Iterate through each JSON file in the specified directory
    for json_file in os.listdir(jsons_directory):
        if json_file.endswith('.json'):
            json_path = os.path.join(jsons_directory, json_file)

            # Extract NCT ID from the JSON file name
            nct_id = json_file.split('.')[0]
            # print(nct_id)
            # Only perform processing if the filename exists in both repositories
            if nct_id in dataframe_filenames:
                # Load existing JSON file
                with open(json_path, 'r') as f:
                    existing_json_data = json.load(f)

                # Load DataFrame for the corresponding NCT ID
                df_path = os.path.join(dataframes_directory, f'{nct_id}.csv')
                df = pd.read_csv(df_path)

                # Group entities by 'text' within each document
                grouped_entities = df.groupby(['nct_id', 'text'])

                # Iterate over each group and update the existing JSON data
                existing_json_data['eligibility'] = []
                for (_, _), group_df in grouped_entities:
                    # Convert the group DataFrame to a list of dictionaries
                    entities_data = group_df[['entity_text', 'entity_group', 'normalized_id', 'field', 'is_negated']].to_dict(orient='records')

                    # Append a dictionary for each 'text'
                    existing_json_data['eligibility'].append({
                        'text': group_df['text'].iloc[0],
                        'entities_data': entities_data
                    })

                # Save the updated JSON data back to the file
                with open(json_path, 'w') as f:
                    json.dump(existing_json_data, f, indent=2)
                
                # Convert the single dictionary into a list before inserting into MongoDB
                collection.insert_many([existing_json_data])

if __name__ == "__main__":
    # Example usage:
    jsons_directory_path = '../data/trials_jsons'
    dataframes_directory_path = '../data/ner_clinical_trials'
    update_json_from_dataframe_and_add_to_mongoDB(jsons_directory_path, dataframes_directory_path)
