import json
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from pydantic import BaseModel, Field
import re

load_dotenv()


# Define the schema for structured output
class PatientStory(BaseModel):
    condition: Optional[str] = Field(
        default=None, description="The main condition of the patient."
    )
    synonyms: Optional[List[str]] = Field(
        default=None, description="Synonyms or related terms for the main condition."
    )
    age: Optional[str] = Field(default=None, description="The age of the patient.")
    gender: Optional[str] = Field(
        default=None, description="The gender of the patient."
    )
    meaningful_sentences: List[str] = Field(
        description="A list of factual, meaningful sentences describing the patient's conditions and entities."
    )


# Initialize OpenAI LLM
llm = ChatOpenAI(
    model=os.environ.get("UMGPT_MODEL", "gpt-4o-mini"),
    temperature=0.5,
    top_p=0.9,
    openai_api_key=os.environ["UMGPT_API_KEY"],
    openai_api_base=os.environ["UMGPT_BASE_URL"],
)


# Function to extract age and gender from the raw description
def extract_age_and_gender(description: str) -> Dict[str, Optional[str]]:
    """
    Extracts age and gender information from the patient description.
    """
    age = None
    gender = None

    # Regex patterns for extracting age and gender
    age_pattern = r"(\b\d{1,3}\b)-?(year-old|yr-old|years old)"
    gender_pattern = r"\b(male|female|man|woman|boy|girl|gentleman|lady)\b"

    age_match = re.search(age_pattern, description, re.IGNORECASE)
    gender_match = re.search(gender_pattern, description, re.IGNORECASE)

    if age_match:
        age = age_match.group(1)
    if gender_match:
        gender = gender_match.group(0).lower()

    # Normalize gender
    if gender in {"man", "male", "boy", "gentleman"}:
        gender = "male"
    elif gender in {"woman", "female", "girl", "lady"}:
        gender = "female"

    return {"age": age, "gender": gender}


# Function to create meaningful sentences for entities
def generate_sentences(description: str, entities: List[str]) -> List[str]:
    """
    Generates meaningful sentences for each entity in the description.
    """
    prompt = f"""
    You are a highly skilled medical assistant tasked with rewriting patient descriptions into factual, descriptive narratives. Each entity provided must be described in its own meaningful sentence. The sentences should strictly describe the patient and avoid making inferences, assumptions, or suggestions.

    Patient Description:
    {description}

    Entities:
    {json.dumps(entities)}

    Guidelines for writing the sentences:
    - Each sentence must describe one entity factually, based on the information provided in the description or the entities list.
    - Avoid making inferences, or suggesting potential outcomes, or suggesting improvements (e.g., no phrases like 'critical for improving' or 'should focus on').
    - Maintain a formal, clinical tone suitable for medical documentation.
    - Ensure statements look as they are written by a medical professional in an official medical record.
    - Ensure that the narrative remains coherent and logical when all sentences are read together. As if they are part of a single patient summary.
    - Ensure to adhere to the description when describing the patient without adding any new information not found in the description.

    Provide the output as a JSON object with the key 'meaningful_sentences' containing the list of sentences.
    """
    response = llm.invoke(
        [HumanMessage(content=prompt)]
    )  # Use invoke for LangChain models
    try:
        # Clean Markdown formatting if present
        structured_output = (
            response.content.strip().strip("```json").strip("```").strip()
        )
        return PatientStory.parse_raw(structured_output).meaningful_sentences
    except Exception as e:
        print(f"Error parsing response: {e}")
        return [f"Error generating sentences for description: {description}"]


# Function to prompt the model to extract the patient's age and gender
def prompt_extract_age_and_gender(description: str) -> Dict[str, Optional[str]]:
    """
    Uses the model to extract age and gender information from the patient description.
    """
    prompt = f"""
    You are a highly skilled medical assistant tasked with identifying specific information from patient descriptions.
    
    Patient Description:
    {description}
    
    Extract the following information:
    - Age of the patient: Explicitly mention the age if it is present in the description.
    - Gender of the patient: Extract gender if it is stated in the description.
    - Normalize the Age to an integer number and Gender to either Male or Female
    - If not mentioned at all in the description, provide 'None' for both age and gender.
    
    Provide the output as a JSON object with the keys 'age' and 'gender'.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        # Parse the JSON response from the model
        structured_output = (
            response.content.strip().strip("```json").strip("```").strip()
        )
        extracted_info = json.loads(structured_output)
        return {
            "age": extracted_info.get("age"),
            "gender": extracted_info.get("gender"),
        }
    except Exception as e:
        print(f"Error extracting age and gender: {e}")
        return {"age": None, "gender": None}


def prompt_extract_main_condition(description: str) -> Dict[str, Optional[str]]:
    """
    Uses the model to extract the main condition from the patient description.
    """
    prompt = f"""
    You are a highly skilled medical assistant tasked with identifying the main condition/disease from patient descriptions.
    
    Patient Description:
    {description}
    
    Extract the main condition mentioned in the description.
    - The main condition should be a specific medical condition or disease that the patient suffers from.
    - If multiple conditions are mentioned, extract the most prominent or relevant one that describes the patient.
    - Ensure the extracted condition is factual and directly related to the patient's health.
    - If no condition is mentioned, provide 'None' as the output.
    - Provide a list of 10 well known aliases or synonyms or related terms if the condition can be described in multiple ways.
 
    Provide the output as a JSON object with the keys 'condition' and 'synonyms'.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        # Parse the JSON response from the model
        structured_output = (
            response.content.strip().strip("```json").strip("```").strip()
        )
        extracted_info = json.loads(structured_output)
        return {
            "condition": extracted_info.get("condition"),
            "synonyms": extracted_info.get("synonyms"),
        }
    except Exception as e:
        print(f"Error extracting main condition: {e}")
        return {"condition": None, "synonyms": None}


# Update the process_file function to use the new age and gender prompt
def process_file_with_prompt(input_file: str, output_file: str):
    """
    Reads patient data from the input file, prompts the model to extract age and gender,
    generates expanded sentences for each patient's entities, and writes the output to the output file.
    """
    with open(input_file, "r") as file:
        data = json.load(file)

    results = {}
    for patient_id, patient_data in data.items():
        print(f"Processing patient {patient_id}...")
        raw_description = patient_data.get("raw", "")
        conditions = patient_data.get("gpt-4-turbo", {}).get("conditions", [])

        # Prompt the model to extract age and gender
        age_gender_info = prompt_extract_age_and_gender(raw_description)

        # Generate meaningful sentences
        expanded_sentences = generate_sentences(raw_description, conditions)

        # Get the main condition
        main_condition_info = prompt_extract_main_condition(raw_description)

        # Save the processed result
        results[patient_id] = {
            "raw_description": raw_description,
            "age": age_gender_info.get("age"),
            "gender": age_gender_info.get("gender"),
            "main_condition": main_condition_info.get("condition"),
            "synonyms": main_condition_info.get("synonyms"),
            "conditions": conditions,
            "expanded_sentences": expanded_sentences,
        }

    # Write the results to the output file
    with open(output_file, "w") as file:
        json.dump(results, file, indent=2)
    print(f"Processing complete. Results saved to {output_file}")


# Main script
if __name__ == "__main__":
    input_file = (
        "../../data/id2queries21.json"  # Replace with the path to your input JSON file
    )
    output_file = "processed_patients21.json"  # Path to save the processed output
    process_file_with_prompt(input_file, output_file)
