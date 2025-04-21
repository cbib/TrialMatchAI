import requests
import xml.etree.ElementTree as ET
import os
import time
import json
import re
import gzip, tarfile

def normalize_whitespace(s):
    return ' '.join(s.split())

def download_study_info(nct_id, runs=2):
    local_file_path = f"../data/trials_xmls/{nct_id}.xml"
    updated_cts = []
    for _ in range(runs):
        if os.path.exists(local_file_path):
            # Read the content of the existing local XML file
            with open(local_file_path, "r") as f:
                local_xml_content = f.read()
            try:
                local_root = ET.fromstring(local_xml_content)
            except ET.ParseError as e:
                print(f"Error parsing XML for trial {nct_id}: {e}")
                os.remove(local_file_path)
                continue
            
            # Download the online version of the XML
            url = f"https://clinicaltrials.gov/ct2/show/{nct_id}?displayxml=true"
            response = requests.get(url)
            
            if response.status_code == 200:
                online_xml_content = response.text
                # Parse the XML content
                online_root = ET.fromstring(online_xml_content)
                to_check = ["eligibility", "brief_title", "overall_status", "location"]
                
                local_version = []
                online_version = []
                
                for s in to_check:
                    local_elem = local_root.find(".//%s" % s)
                    online_elem = online_root.find(".//%s" % s)
                    
                    # Check if the element exists in both versions
                    if local_elem is not None and online_elem is not None:
                        local_version.append(local_elem)
                        online_version.append(online_elem)
                    else:
                        continue
                
                is_updated = any([normalize_whitespace(ET.tostring(a, encoding='unicode').strip()) !=
                                normalize_whitespace(ET.tostring(b, encoding='unicode').strip())
                                for a, b in zip(local_version, online_version)])
                
                if is_updated:
                    updated_cts.append(nct_id)
                    # Update the local XML with the online version
                    with open(local_file_path, "w") as f:
                        f.write(ET.tostring(online_root, encoding='unicode'))
                    print(f"Updated eligibility criteria for {nct_id}")
                else:
                    print(f"No changes in eligibility criteria for {nct_id}.")
            else:
                print(f"Error downloading study information for {nct_id}")
        else:
            downloaded = False
            while not downloaded:
                url = f"https://clinicaltrials.gov/ct2/show/{nct_id}?displayxml=true"
                response = requests.get(url)
                if response.status_code == 200:
                    root = ET.fromstring(response.text)
                    with open(local_file_path, "w") as f:
                        f.write(ET.tostring(root, encoding='unicode'))
                    downloaded = True
                    print(f"Study information downloaded for {nct_id}")
                else:
                    print(f"Error downloading study information for {nct_id}")
                
                if not downloaded:
                    print(f'Download of {nct_id}.xml failed. Retrying in 2 seconds...')
                    time.sleep(2)
    return updated_cts


def extract_study_info(nct_id):
    """
    Extract various study information from a clinical trial text with the given NCT identifier.

    This function attempts to extract various study information for a clinical trial specified by its unique
    NCT identifier (NCT ID). The function checks if a file named '{nct_id}_info.txt' already exists
    in the 'trials_texts' directory. If the file exists, the function returns 0, indicating that the
    extraction is not required, and the information is already available locally.

    If the file '{nct_id}_info.txt' does not exist, the function parses the XML file with the name '{nct_id}.xml'
    located in the 'trials_texts' directory. The XML content is parsed using the `xml.etree.ElementTree`
    module. The function then extracts various study information from the XML content and saves it in a text file
    with the name '{nct_id}_info.txt' in the 'trials_texts' directory.

    The extracted study information includes:
    - Long title
    - Short title
    - Cancer sites
    - Start date
    - End date
    - Primary end date
    - Overall status
    - Study phase
    - Study type
    - Brief summary
    - Detailed description
    - Number of arms
    - Arms information
    - Eligibility criteria
    - Gender
    - Minimum age
    - Maximum age
    - Intervention details
    - Location details

    Parameters:
        nct_id (str): The unique identifier (NCT ID) of the clinical trial for which study information
                    needs to be extracted.

    Returns:
        int: Returns 0 if the study information file already exists locally and doesn't require extraction.
            Otherwise, the function doesn't return anything directly (implicit return).
            Note: The extracted study information is saved in the 'trials_texts' directory.

    """
    if os.path.exists(f"../data/trials_xmls/{nct_id}_info.txt"):
        return 0
        # print(f"{nct_id}_info.txt already exists. Skipping extraction.")
    else:
        tree = ET.parse(f"../data/trials_xmls/{nct_id}.xml")
        root = tree.getroot()
        with open(f"../data/trials_xmls/{nct_id}_info.txt", "w") as f:
            
            # Extract Long title
            official_title = root.find(".//official_title")
            if official_title is not None:
                title_text = official_title.text.strip()
                f.write(f"Long Title:\n{title_text}\n\n")
                
            # Extract short title
            brief_title = root.find(".//brief_title")
            if brief_title is not None:
                title_text = brief_title.text.strip()
                f.write(f"Short Title:\n{title_text}\n\n")
            
            # Extract cancer sites
            conditions = root.findall(".//condition")
            if conditions is not None:
                f.write("Cancer Site(s):\n")
                for condition in conditions:
                    condition_text = condition.text.strip()
                    f.write(f"- {condition_text}\n")
                f.write("\n")

            # Extract start date
            start_date = root.find(".//start_date")
            if start_date is not None:
                start_date_text = start_date.text.strip()
                f.write(f"Start Date:\n{start_date_text}\n\n")

            # Extract end date
            end_date = root.find(".//completion_date")
            if end_date is not None:
                end_date_text = end_date.text.strip()
                f.write(f"End Date:\n{end_date_text}\n\n")
                
            # Extract primary end date
            primary_end_date = root.find(".//primary_completion_date")
            if end_date is not None:
                end_date_text = end_date.text.strip()
                f.write(f"Primary End Date:\n{end_date_text}\n\n")
            
            
            # Extract overall status
            overall_status = root.find(".//overall_status")
            if overall_status is not None:
                overall_status_text = overall_status.text.strip()
                f.write(f"Overall Status:\n{overall_status_text}\n\n")
                
            # Extract study phase
            study_phase = root.find(".//phase")
            if study_phase is not None:
                f.write(f"Study Phase: \n{study_phase.text.strip()}\n\n")

            # Extract study type
            study_type = root.find(".//study_type")
            if study_type is not None:
                study_type_text = study_type.text.strip()
                f.write(f"Study Type:\n{study_type_text}\n\n")
                
            # Extract brief summary
            brief_summary = root.find(".//brief_summary")
            if brief_summary is not None:
                brief_summary_text = brief_summary.find(".//textblock").text.strip()
                f.write(f"Brief Summary:\n{brief_summary_text}\n\n")
                
            # Extract detailed description
            detailed_description = root.find(".//detailed_description")
            if detailed_description is not None:
                detailed_description_text = detailed_description.find(".//textblock").text.strip()
                f.write(f"Detailed Description:\n{detailed_description_text}\n\n")
                
            # Extract number of arms
            number_of_arms = root.find(".//number_of_arms")
            if number_of_arms is not None:
                f.write(f"Number of Arms: {number_of_arms.text.strip()}\n\n")

            arms = root.findall(".//arm_group")
            if arms is not None:
                f.write("Arms:\n")
                for arm in arms:
                    arm_group_label = arm.find(".//arm_group_label").text.strip()
                    arm_group_description = arm.find(".//arm_group_description")
                    if arm_group_description is not None:
                        arm_group_description_text = arm_group_description.text.strip()
                        f.write(f"- {arm_group_label}: {arm_group_description_text}\n")
                    else:
                        f.write(f"- {arm_group_label}\n")
                f.write("\n")
            
            # Extract eligibility criteria
            eligibility_criteria = root.find(".//eligibility/criteria")
            if eligibility_criteria is not None:
                eligibility_criteria_text = eligibility_criteria.find(".//textblock").text.strip()
                f.write(f"Eligibility Criteria:\n{eligibility_criteria_text}\n\n")

            # Extract gender
            gender = root.find(".//gender")
            if gender is not None:
                gender_text = gender.text.strip()
                f.write(f"Gender:\n{gender_text}\n\n")

            # Extract minimum age
            min_age = root.find(".//eligibility/minimum_age")
            if min_age is not None:
                min_age_text = min_age.text.strip()
                f.write(f"Minimum Age:\n{min_age_text}\n\n")
            
            # Extract maximum age
            max_age = root.find(".//eligibility/maximum_age")
            if max_age is not None:
                max_age_text = max_age.text.strip()
                f.write(f"Maximum Age:\n{max_age_text}\n\n")

            # Extract intervention
            intervention = root.findall(".//intervention")
            if intervention is not None:
                f.write("Interventions:\n")
                for i in intervention:
                    intervention_name = i.find(".//intervention_name").text.strip()
                    f.write(f"- {intervention_name}\n")
                f.write("\n")
                
            # Extract locations
            locations = root.findall(".//location")
            if locations is not None:
                f.write("Locations:\n")
                for location in locations:
                    city = location.find(".//city")
                    country = location.find(".//country")
                    if city is not None and country is not None:
                        location_text = f"{city.text.strip()}, {country.text.strip()}"
                        f.write(f"- {location_text}\n")
                f.write("\n")

    print(f"{nct_id} info extracted and saved to {nct_id}_info.txt")

def add_spaces_around_punctuation(text):

    """
    Add spaces around punctuation

    Parameters
    ----------
    text : str
        The text to be preprocessed

    Returns
    -------
    str
        The preprocessed text
    """
    text = re.sub(r'([.,!?()])', r' \1 ', text)
    return text


def remove_special_characters(text):
    """
    Remove special characters

    Parameters
    ----------
    text : str
        The text to be preprocessed

    Returns
    -------
    str
        The preprocessed text
    """
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    return text

def remove_dashes_at_the_start_of_sentences(text):
    """
    Remove dashes at the start of sentences

    Parameters
    ----------
    text : str
        The text to be preprocessed

    Returns
    -------
    str
        The preprocessed text
    """
    text = re.sub(r'^- ', '', text)
    return text


def post_process_entities(entities):
    """
    Merge consecutive entities and post-process the results.

    This function takes a list of entities generated from a named entity recognition (NER) model's output
    and performs post-processing to merge consecutive entities of the same type. The input entities list
    contains dictionaries representing each detected entity with the following keys:
    - "entity" (str): The entity type represented as a prefixed tag (e.g., "B-ORG", "I-LOC").
    - "score" (float): The confidence score assigned to the entity by the NER model.
    - "word" (str): The text of the entity in the input text.
    - "start" (int): The starting index of the entity in the input text.
    - "end" (int): The ending index (exclusive) of the entity in the input text.

    The function iterates through the entities and merges consecutive entities with the same type into a single
    entity. It also handles entities that span multiple words, indicated by the presence of "I-" prefixes.
    The merged entity is represented by a dictionary containing the merged information:
    - "entity" (str): The entity type without the prefix (e.g., "ORG", "LOC").
    - "score" (float): The maximum confidence score among the merged entities.
    - "word" (str): The combined text of the merged entities.
    - "start" (int): The starting index of the first entity in the merged sequence.
    - "end" (int): The ending index (exclusive) of the last entity in the merged sequence.

    Parameters:
        entities (list): A list of dictionaries representing detected entities.

    Returns:
        list: A list of dictionaries representing merged entities after post-processing.
            Each dictionary contains the keys "entity", "score", "word", "start", and "end"
            representing the entity type, confidence score, text, start index, and end index respectively.
    """
    merged_entities = []
    current_entity = None

    for entity in entities:
        if entity["entity"].startswith("B-"):
            if current_entity is not None:
                merged_entities.append(current_entity)
            current_entity = {
                "entity": entity["entity"][2:],
                "score": entity["score"],
                "word": entity["word"].replace("##", " "),
                "start": entity["start"],
                "end": entity["end"]
            }
        elif entity["entity"].startswith("I-"):
            if (current_entity is not None) and entity["word"].startswith("##"):
                current_entity["word"] += entity["word"].replace("##", "")
                current_entity["end"] = entity["end"]
                current_entity["score"] = max(current_entity["score"], entity["score"])
            else:
                current_entity["word"] += " " + entity["word"].lstrip()
                current_entity["end"] = entity["end"]
                current_entity["score"] = max(current_entity["score"], entity["score"])
        else:
            if current_entity is not None:
                merged_entities.append(current_entity)
                current_entity = None

    if current_entity is not None:
        merged_entities.append(current_entity)

    return merged_entities


def get_dictionaries_with_values(list_of_dicts, key, values):
    """
    Filter a list of dictionaries based on the presence of specific values in a specified key.

    This function takes a list of dictionaries and filters them based on the presence of specific values in a specified key.
    The function checks each dictionary in the input list and includes only those dictionaries where any of the given values
    are present in the specified key. The filtering is performed using list comprehensions.

    Parameters:
        list_of_dicts (list): A list of dictionaries to be filtered.
        key (str): The key in the dictionaries where the filtering is applied.
        values (list): A list of values. The function will filter dictionaries where any of these values are present in the specified key.

    Returns:
        list: A list of dictionaries that meet the filtering criteria.

    Example:
        list_of_dicts = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
            {"name": "Charlie", "age": 35},
            {"name": "David", "age": 30},
        ]

        get_dictionaries_with_values(list_of_dicts, "age", [30, 35])
        # Output: [
        #   {"name": "Alice", "age": 30},
        #   {"name": "Charlie", "age": 35},
        #   {"name": "David", "age": 30}
        # ]
    """
    return [d for d in list_of_dicts if any(val in d.get(key, []) for val in values)]

def resolve_ner_overlaps(ner1_results, ner2_results):
    """
    Resolve overlaps between entities detected by two named entity recognition (NER) models.

    This function takes the results of two NER models (ner1_results and ner2_results) and resolves overlaps
    between the entities detected by these models. An overlap occurs when the span of an entity detected by one
    model partially or fully overlaps with the span of an entity detected by the other model.

    The function iterates through the entities detected by the first NER model (ner1_results). For each entity,
    it checks if it overlaps with any entity from the second model (ner2_results). If there are no overlaps,
    the entity from the first model is added to the resolved results.

    After processing the entities from the first model, the function then adds entities from the second model
    that do not overlap with any entities from the first model.

    Parameters:
        ner1_results (list): A list of dictionaries representing entities detected by the first NER model.
        ner2_results (list): A list of dictionaries representing entities detected by the second NER model.

    Returns:
        list: A list of dictionaries representing the resolved entities with overlaps removed.

    Example:
        ner1_results = [
            {"start": 5, "end": 10, "entity_group": "PERSON"},
            {"start": 20, "end": 25, "entity_group": "LOCATION"}
        ]

        ner2_results = [
            {"start": 8, "end": 15, "entity_group": "PERSON"},
            {"start": 18, "end": 30, "entity_group": "ORGANIZATION"}
        ]

        resolve_ner_overlaps(ner1_results, ner2_results)
        # Output: [
        #   {"start": 5, "end": 10, "entity_group": "PERSON"},
        #   {"start": 18, "end": 30, "entity_group": "ORGANIZATION"},
        #   {"start": 20, "end": 25, "entity_group": "LOCATION"}
        # ]
    """
    resolved_results = []
    # Iterate over the entities detected by the first NER model
    for entity1 in ner1_results:
        entity1_start = entity1['start']
        entity1_end = entity1['end']
        entity1_label = entity1['entity_group']

        # Check if the entity from the first model overlaps with any entity from the second model
        overlaps = False
        for entity2 in ner2_results:
            entity2_start = entity2['start']
            entity2_end = entity2['end']
            entity2_label = entity2['entity_group']

            if entity1_start < entity2_end and entity1_end > entity2_start:
                overlaps = True
                break

        # If there were no overlaps, add the entity from the first model to the resolved results
        if not overlaps:
            resolved_results.append(entity1)

    # Add entities from the second model that don't overlap with any entities from the first model
    for entity2 in ner2_results:
        entity2_start = entity2['start']
        entity2_end = entity2['end']
        entity2_label = entity2['entity_group']

        overlaps = False
        for entity1 in resolved_results:
            entity1_start = entity1['start']
            entity1_end = entity1['end']
            entity1_label = entity1['entity_group']

            if entity2_start < entity1_end and entity2_end > entity1_start:
                overlaps = True
                break

        if not overlaps:
            resolved_results.append(entity2)

    return resolved_results

def extract_eligibility_criteria(trial_id):
    """
    Extract the eligibility criteria text for a clinical trial with the given trial ID.

    This function attempts to locate and extract the eligibility criteria text for a clinical trial
    specified by its trial ID. The function reads an XML file named '{trial_id}.xml' which is expected
    to contain information for the clinical trial. It searches for the eligibility criteria textblock within
    the XML and extracts the corresponding text.

    Parameters:
        trial_id (str): The unique identifier of the clinical trial.

    Returns:
        str or None: The extracted eligibility criteria text for the specified trial if found,
                    otherwise None.
    """
    xml_file_path = f'../data/trials_xmls/{trial_id}.xml'

    if os.path.exists(xml_file_path):
        with open(xml_file_path, 'r') as xml_file:
            xml_content = xml_file.read()
        try:
            tree = ET.ElementTree(ET.fromstring(xml_content))
            root = tree.getroot()
        except ET.ParseError as e:
            print(f"Error parsing XML for trial {trial_id}: {e}")
            return None
        # Find the Eligibility Criteria TextBlock section within the XML
        eligibility_criteria_textblock = root.find(".//eligibility/criteria/textblock")

        if eligibility_criteria_textblock is not None:
            # Extract the text from the Eligibility Criteria TextBlock section
            eligibility_criteria_text = eligibility_criteria_textblock.text
            return eligibility_criteria_text.strip()

    # If the trial ID is not found or the eligibility criteria textblock is missing, return None
    return None



def replace_parentheses_with_braces(text):
    """
    Replace parentheses with curly braces in the given text.

    This function takes a text as input and replaces all occurrences of opening parentheses '('
    with an opening curly brace '{', and closing parentheses ')' with a closing curly brace '}'.
    The function maintains a stack to ensure proper matching of parentheses. If a closing parenthesis
    is encountered without a corresponding opening parenthesis in the stack, it is left unchanged.

    Parameters:
        text (str): The input text containing parentheses that need to be replaced.

    Returns:
        str: The modified text with parentheses replaced by curly braces.
    """
    stack = []
    result = ""
    for char in text:
        if char == '(' or char == '[':
            stack.append(char)
            result += "{"
        elif char == ')' or char == "]":
            if stack:
                stack.pop()
                result += "}"
            else:
                result += char
        else:
            result += char
    return result




def line_starts_with_capitalized_alphanumeric(line):
    """
    Check if the given line starts with a capitalized alphanumeric word.

    Parameters:
        line (str): The input string representing a line.

    Returns:
        bool: True if the line starts with a capitalized alphanumeric word, False otherwise.
    """
    words = line.split()
    if len(words) > 0:
        first_word = words[0]
        if first_word[0].isalpha() and first_word[0].isupper():
            return True
    return False

