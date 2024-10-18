# Install necessary packages
# pip install langchain transformers accelerate sentencepiece

from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Choose the model you want to use
model_name = "HuggingFaceH4/zephyr-7b-beta"  # Or any other compatible model

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=False,  # Some models may not have a fast tokenizer
    trust_remote_code=True  # Needed for some models with custom code
)

# Load the model with appropriate device settings
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use float16 for better performance if GPU supports it
    device_map="auto",  # Automatically assigns layers to devices
    trust_remote_code=True
)

# Create a text-generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0.7,
    do_sample=True,
    top_p=0.95,
    repetition_penalty=1.2,
    pad_token_id=tokenizer.eos_token_id,  # Avoids tokenizer errors
    truncation=True
)

# Initialize the HuggingFacePipeline LLM
llm = HuggingFacePipeline(pipeline=pipe)

# Create a prompt template for generating the summary
template = """
You are a medical assistant helping to determine if a patient is a good fit for a clinical trial.

**Clinical Trial Eligibility Criteria:**
{eligibility_criteria}

**Patient Profile:**
{patient_profile}

Based on the above information, provide a detailed summary explaining why the patient is a good fit for the clinical trial. Include any relevant details and considerations.

**Summary:**
"""

prompt = PromptTemplate(
    input_variables=["eligibility_criteria", "patient_profile"],
    template=template,
)

# Create an LLMChain with the prompt and LLM
chain = LLMChain(llm=llm, prompt=prompt)

# Function to generate the summary
def generate_summary(eligibility_criteria, patient_profile):
    return chain.run(eligibility_criteria=eligibility_criteria, patient_profile=patient_profile).strip()

# Example usage
if __name__ == "__main__":
    # Example clinical trial eligibility criteria
    eligibility_criteria = """
    - Age between 18 and 65 years.
    - Diagnosed with Type 2 Diabetes Mellitus for at least 1 year.
    - HbA1c level between 7.0% and 10.0%.
    - Body Mass Index (BMI) between 25 and 40 kg/m².
    - Not currently on insulin therapy.
    - No history of cardiovascular disease.
    """

    # Example patient profile
    patient_profile = """
    - Age: 45 years old.
    - Diagnosis: Type 2 Diabetes Mellitus diagnosed 5 years ago.
    - HbA1c level: 8.5%.
    - BMI: 30 kg/m².
    - Current medications: Metformin.
    - Medical history: No cardiovascular disease.
    - Lifestyle: Non-smoker, engages in moderate exercise.
    """

    summary = generate_summary(eligibility_criteria, patient_profile)
    print("Summary:")
    print(summary)




# Install necessary packages
# pip install langchain openai

from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain

# For secure API key handling
import os

# Initialize the OpenAI Chat LLM
llm = ChatOpenAI(
    openai_api_key='',  # Replace with your actual API key
    temperature=0.7,  # Controls creativity; adjust between 0 and 1
    model_name='gpt-4o-mini-2024-07-18'  # You can specify 'gpt-4' if you have access
)
# Create a prompt template for generating the summary
template = """
You are a medical assistant helping to determine if a patient is a good fit for a clinical trial.

**Clinical Trial Eligibility Criteria:**
{eligibility_criteria}

**Patient Profile:**
{patient_profile}

Based on the above information, provide a detailed summary explaining why the patient is a good fit for the clinical trial.

**Summary:**
"""

prompt = PromptTemplate(
    input_variables=["eligibility_criteria", "patient_profile"],
    template=template,
)

# Create an LLMChain with the prompt and LLM
chain = LLMChain(llm=llm, prompt=prompt)

# Function to generate the summary
def generate_summary(eligibility_criteria, patient_profile):
    return chain.run(eligibility_criteria=eligibility_criteria, patient_profile=patient_profile).strip()

# Example usage
if __name__ == "__main__":
    # Example clinical trial eligibility criteria
    eligibility_criteria = """
    - Age between 18 and 65 years.
    - Diagnosed with Type 2 Diabetes Mellitus for at least 1 year.
    - HbA1c level between 7.0% and 10.0%.
    - Body Mass Index (BMI) between 25 and 40 kg/m².
    - Not currently on insulin therapy.
    - No history of cardiovascular disease.
    """

    # Example patient profile
    patient_profile = """
    - Age: 45 years old.
    - Diagnosis: Type 2 Diabetes Mellitus diagnosed 5 years ago.
    - HbA1c level: 8.5%.
    - BMI: 30 kg/m².
    - Current medications: Metformin.
    - Medical history: No cardiovascular disease.
    - Lifestyle: Non-smoker, engages in moderate exercise.
    """

    summary = generate_summary(eligibility_criteria, patient_profile)
    print("Summary:")
    print(summary)
