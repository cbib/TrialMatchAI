from entity_recognition import EntityRecognizer

import pandas as pd
import os
folder_path = '../data/trials_xmls/'  # Replace this with the path to your folder
file_names = []
# List all files in the folder
for file in os.listdir(folder_path):
    if os.path.isfile(os.path.join(folder_path, file)):
        file_name, file_extension = os.path.splitext(file)
        file_names.append(file_name)

reco = EntityRecognizer(n_jobs=50, id_list=file_names, data_source="clinical trials")

entities = reco()