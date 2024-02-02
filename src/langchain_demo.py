from langchain.llms import OpenAI
from langchain import PromptTemplate
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from numpy.linalg import norm
import requests
import openai
import pymongo
import os
import json
import numpy as np
import pandas as pd
from connect_to_mongodb import connect_mongodb

# Set OpenAI API Key
import os

# Replace 'your_api_key_here' with your actual OpenAI API key
openai_api_key = 'your_api_key_here'

# Set OpenAI API key as an environment variable
os.environ['OPENAI_API_KEY'] = openai_api_key

# Define the URL
url = "https://en.wikipedia.org/wiki/Markets_in_Financial_Instruments_Directive_2014#Directive_2014/65/EU_/_Regulation_(EU)_No_600/2014"

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']

def data_prep():
    # Load data
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text = soup.get_text()
    content_div = soup.find('div', {'class': 'mw-parser-output'})

    # Remove unwanted elements from div
    unwanted_tags = ['sup', 'span', 'table', 'ul', 'ol']
    for tag in unwanted_tags:
        for match in content_div.findAll(tag):
            match.extract()
    article_text = content_div.get_text()

    # Split data
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""],
        length_function=len,
    )
    texts = text_splitter.create_documents([article_text])

    # Calculate embeddings
    text_chunks = [text.page_content for text in texts]
    df = pd.DataFrame({'text_chunks': text_chunks})
    df['ada_embedding'] = df.text_chunks.apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))

    # Store into MongoDB
    collection = connect_mongodb()
    df_dict = df.to_dict(orient='records')
    collection.insert_many(df_dict)

    print("Data loaded successfully")