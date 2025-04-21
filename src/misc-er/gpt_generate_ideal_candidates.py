import os
import json
import random
import re
import ast
from typing import List, Dict, Tuple
from dateutil.parser import parse
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# Set your API key securely in production.
os.environ["OPENAI_API_KEY"] = ""

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", max_retries=3)

# Minimum word count for eligibility criteria to be considered "rich"
MIN_ELIGIBILITY_WORD_COUNT = 512

def safe_parse_list(response_str: str) -> List:
    """
    Attempts to parse a Python list from a string.
    Extracts the content between the first "[" and the last "]".
    First tries ast.literal_eval; if that fails, replaces single quotes with double quotes and tries json.loads.
    Returns an empty list if parsing is unsuccessful.
    """
    start = response_str.find("[")
    end = response_str.rfind("]")
    if start == -1 or end == -1:
        print("No list brackets found in the response.")
        return []
    list_str = response_str[start:end+1]
    
    try:
        result = ast.literal_eval(list_str)
        if isinstance(result, list):
            return result
    except Exception as e:
        print(f"literal_eval failed on extracted string: {e}")
    
    try:
        json_str = list_str.replace("'", '"')
        result = json.loads(json_str)
        if isinstance(result, list):
            return result
    except Exception as e:
        print(f"json.loads fallback failed: {e}")
    
    return []

def safe_parse_dict(response_str: str) -> Dict:
    """
    Attempts to parse a Python dict from a string.
    Extracts the substring between the first "{" and the last "}".
    First tries ast.literal_eval; if that fails, replaces single quotes with double quotes and tries json.loads.
    Returns an empty dict if unsuccessful.
    """
    start = response_str.find("{")
    end = response_str.rfind("}")
    if start == -1 or end == -1:
        print("No dictionary found in the response.")
        return {}
    dict_str = response_str[start:end+1]
    
    try:
        result = ast.literal_eval(dict_str)
        if isinstance(result, dict):
            return result
    except Exception as e:
        print(f"literal_eval failed on extracted string: {e}")
    
    try:
        json_str = dict_str.replace("'", '"')
        result = json.loads(json_str)
        if isinstance(result, dict):
            return result
    except Exception as e:
        print(f"json.loads fallback failed: {e}")
    
    return {}

def generate_synonyms(condition: str) -> List[str]:
    """
    Generates 10 well-known synonyms or alternative names for the given condition.
    The model is instructed to provide the output as a Python list.
    """
    prompt = f"""
    You are a medical expert. Generate 10 well-known synonyms or alternative names for the following condition:
    Condition: {condition}
    Provide the output as a Python list of strings.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    print("Synonyms response:", response.content)
    synonyms = safe_parse_list(response.content)
    if isinstance(synonyms, list):
        synonyms = [str(s) for s in synonyms if isinstance(s, (str, int, float))]
    else:
        synonyms = []
        print(f"Error parsing synonyms for condition '{condition}'.")
    return synonyms

def generate_summary(details: List[str], conditions: List[str] = None, age_spec: str = None) -> str:
    """
    Generates a one-paragraph patient note based on the provided details and conditions.
    The note should naturally incorporate the necessary patient information without explicitly referencing
    eligibility criteria, trial requirements, or any commentary on meeting specific conditions.
    If an age_spec is provided, include this information in the note. The note must end with a separate
    line exactly in the format:
    "Age: <number>, Gender: <value>"
    """
    combined_details = " ".join(details)
    
    condition_info = ""
    if conditions:
        condition_info = "diagnosed with one of : " + ", ".join(conditions) + "."
    
    age_sentence = f" The patient should be {age_spec}." if age_spec else ""
    
    prompt = f"""
You are a seasoned medical expert. Based on the clinical trial information provided below, generate a detailed, professional one-paragraph patient note describing an ideal candidate {condition_info} for the trial. Ensure that the candidate's medical history and current condition strictly satisfy every single inclusion criterion without exception while clearly and explicitly not violating any exclusion criteria.

Age and gender requirements for the trial: {age_sentence}

Eligibility Details: {combined_details}

Please integrate the above information naturally into the note as if describing the patient's history and presentation, without any explicit reference to suitability of the patient with the provided eligibility criteria, trial requirements, or statements like "meets the criterion" or "satisfies the requirements of the trial" etc... Your note should simply convey the patient's condition and background in a realistic manner similar to an EHR or admission note.
Go over the criteria one-by-one. They should all be covered in the patient's note without exception.

At the end of the note, include a new line exactly in the following format:
"Age: <number>, Gender: <value>"
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    print("Summary response:", response.content)
    return response.content.strip()

def extract_age_gender_from_summary(summary: str) -> Tuple[int, str]:
    """
    Extracts the age and gender from the summary using a regular expression.
    Expects the summary to end with a line in the format:
    "Age: <number>, Gender: <value>"
    """
    pattern = r"Age:\s*(\d+)\s*,\s*Gender:\s*([A-Za-z]+)"
    match = re.search(pattern, summary)
    if match:
        age = int(match.group(1))
        gender = match.group(2).lower()
        if gender not in ["male", "female"]:
            gender = "male"
        return age, gender
    else:
        print("Could not extract age and gender from summary. Using default values.")
        return 50, "male"

def split_into_sentences(text: str) -> List[str]:
    """
    Splits the provided text into sentences using punctuation as delimiters.
    """
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if s.strip()]

def generate_conditions(raw_description: str) -> List[str]:
    """
    Extracts all relevant medical conditions solely based on the patient’s raw description.
    The model is instructed to provide the output as a Python list of strings.
    """
    prompt = f"""
    You are a medical expert. Based solely on the following patient note, extract all relevant medical conditions mentioned.
    Provide the output as a Python list of strings.
    Patient Note: {raw_description}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    print("Conditions response:", response.content)
    conditions = safe_parse_list(response.content)
    if isinstance(conditions, list):
        conditions = [str(c) for c in conditions if isinstance(c, (str, int, float))]
    else:
        conditions = []
        print("Error parsing conditions.")
    return conditions

def extract_main_condition_from_summary(summary: str) -> str:
    """
    Extracts the primary condition for which the patient is being treated from the patient note.
    The model is instructed to return a concise, single phrase.
    """
    prompt = f"""
    You are a medical expert. Based solely on the following patient note, identify the primary condition for which the patient is being treated. Provide only a concise, single phrase.
    Patient Note: {summary}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    main_condition = response.content.strip()
    print("Extracted main condition:", main_condition)
    return main_condition

def extract_age_gender_from_trial(trial_data: Dict) -> str:
    """
    Extracts the age and gender requirements from the trial's JSON data using the fields:
    "minumum_age", "maximum_age", and "gender".
    Constructs and returns an age specification string.
    """
    try:
        min_age_raw = trial_data.get("minumum_age")
        max_age_raw = trial_data.get("maximum_age")
        trial_gender = trial_data.get("gender", "male")
        if min_age_raw is not None and max_age_raw is not None:
            if isinstance(min_age_raw, str):
                min_age = int(''.join(filter(str.isdigit, min_age_raw)))
            else:
                min_age = int(min_age_raw)
            if isinstance(max_age_raw, str):
                max_age = int(''.join(filter(str.isdigit, max_age_raw)))
            else:
                max_age = int(max_age_raw)
            age_spec = f"aged between {min_age} and {max_age} years old; gender must be {trial_gender}"
        else:
            age_spec = f"gender must be {trial_gender}"
    except Exception as e:
        print(f"Error constructing age specification: {e}")
        age_spec = f"gender must be {trial_gender}"
    return age_spec

def generate_patient_profile(eligibility_criteria: str, ground_nctid: str, trial_data: Dict) -> Dict:
    """
    Generates a patient profile for a trial.
    
    Steps:
      1. Construct an age specification string from the trial's JSON.
      2. Extract condition(s) from the trial data.
      3. Generate a comprehensive raw patient note (summary) based on the trial's eligibility criteria and conditions.
         The note will be generated using the eligibility criteria, conditions, and will include a final line with age and gender.
      4. Extract the age and gender from the generated note.
      5. Extract the main condition from the note.
      6. Generate synonyms for the extracted main condition.
      7. Extract additional conditions solely from the raw note.
    
    Returns a JSON object with the following keys:
      - raw_description (the patient note)
      - age
      - gender
      - main_condition
      - synonyms
      - conditions
      - split_raw_description (the note split into sentences)
      - ground_nctid
    """
    # 1. Build an age specification string from trial_data.
    age_spec = extract_age_gender_from_trial(trial_data)
    
    # 2. Extract condition(s) from the trial data.
    condition_field = trial_data.get("condition", "")
    if isinstance(condition_field, list):
        conditions_input = condition_field  # use the list directly
    elif isinstance(condition_field, str) and condition_field.strip():
        conditions_input = [condition_field]
    else:
        conditions_input = []
    
    # 3. Generate the raw patient note using eligibility criteria, conditions, and age specifications.
    raw_description = generate_summary([eligibility_criteria], conditions=conditions_input, age_spec=age_spec)
    
    # 4. Extract age and gender from the generated note.
    age, gender = extract_age_gender_from_summary(raw_description)
    
    # 5. Extract the main condition from the raw description.
    main_condition = extract_main_condition_from_summary(raw_description)
    
    # 6. Generate synonyms for the main condition.
    synonyms = generate_synonyms(main_condition)
    
    # 7. Extract additional conditions solely from the raw note.
    conditions_from_note = generate_conditions(raw_description)
    if main_condition not in conditions_from_note:
        conditions = [main_condition] + conditions_from_note
    else:
        conditions = conditions_from_note

    return {
        "raw_description": raw_description,
        "age": age,
        "gender": gender,
        "main_condition": main_condition,
        "synonyms": synonyms,
        "conditions": conditions,
        "split_raw_description": split_into_sentences(raw_description),
        "ground_nctid": ground_nctid
    }

def process_trials(input_folder: str, output_file: str):
    """
    Processes trial JSON files in the given folder:
      1. Searches for files with names starting with "NCT" and ending with ".json".
      2. For each file, checks that the trial's "start_date" is after 2015, that the "condition"
         (when lowercased) contains cancer-related terms (e.g., "cancer", "tumor", "malignancy", "neoplasm", "carcinoma"),
         that the eligibility criteria are sufficiently long and rich, and that the trial's overall_status is "Recruiting".
      3. Randomly samples 100 trials that meet these conditions.
      4. For each trial, uses its "eligibility_criteria" to generate a patient profile that perfectly fits the trial.
         The profile is built by generating a raw patient note (aware of age/gender specs and conditions) and then extracting the main condition,
         synonyms, and additional conditions from that note.
      5. Saves the 100 patient profiles to a single JSON file.
    """
    all_files = [f for f in os.listdir(input_folder) if f.startswith("NCT") and f.endswith(".json")]
    matching_trials = []
    cancer_terms = ["cancer", "tumor", "malignancy", "neoplasm", "carcinoma"]

    for filename in all_files:
        file_path = os.path.join(input_folder, filename)
        try:
            with open(file_path, 'r') as f:
                trial_data = json.load(f)
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
            continue

        # Check if the start_date is after 2015.
        start_date_str = trial_data.get("start_date")
        if not start_date_str or not isinstance(start_date_str, str):
            print(f"Skipping trial {filename} because start_date is missing or not a valid string.")
            continue

        try:
            parsed_date = parse(start_date_str)
            year = parsed_date.year
        except Exception as e:
            print(f"Error parsing start_date in {filename}: {e}")
            year = 0

        # Check if the eligibility criteria exist and are long and rich.
        eligibility_criteria = trial_data.get("eligibility_criteria", "")
        if not eligibility_criteria or len(eligibility_criteria.split()) < MIN_ELIGIBILITY_WORD_COUNT:
            print(f"Skipping trial {filename} due to insufficient eligibility criteria (less than {MIN_ELIGIBILITY_WORD_COUNT} words).")
            continue

        # Handle the condition field, which can be a list or a string.
        condition_field = trial_data.get("condition", "")
        if isinstance(condition_field, list):
            condition_str = " ".join(condition_field)
        else:
            condition_str = condition_field
        condition_lower = condition_str.lower()

        # Additional condition: trial must have overall_status "Recruiting"
        overall_status = trial_data.get("overall_status", "").strip().lower()

        if (year > 2015 and 
            any(term in condition_lower for term in cancer_terms) and 
            overall_status == "recruiting"):
            matching_trials.append((filename, trial_data))
    
    if len(matching_trials) < 100:
        print(f"Warning: Only {len(matching_trials)} matching trials found. Proceeding with available trials.")
        sample_trials = matching_trials
    else:
        sample_trials = random.sample(matching_trials, 100)
    
    results = {}
    for filename, trial_data in sample_trials:
        ground_nctid = filename.replace(".json", "")
        eligibility_criteria = trial_data.get("eligibility_criteria", "")
        print(f"Processing trial {ground_nctid}...")
        patient_profile = generate_patient_profile(eligibility_criteria, ground_nctid, trial_data)
        results[ground_nctid] = patient_profile
    
    with open(output_file, 'w') as out_file:
        json.dump(results, out_file, indent=2)
    
    print(f"Processing complete. Results saved to {output_file}")

# Set your input folder and output file paths accordingly.
input_folder = "../../data/trials_jsons"  # Folder containing NCT*.json files
output_file = "perfect_patient_profiles.json"

process_trials(input_folder, output_file)
