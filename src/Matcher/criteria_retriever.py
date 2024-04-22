import os
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from typing import List, Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_core.runnables import chain

    
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

load_dotenv('../.env')
openai_access_key = os.getenv('OPENAI_ACCESS_KEY')

vectorstore = Chroma(persist_directory="../../data/db", embedding_function= SentenceTransformerEmbeddings(), collection_name="criteria")

retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold", 
    search_kwargs={"score_threshold": 0.5, "k":15000},
    filters=None,
)

class Search(BaseModel):
    """Search over a database of clinical trial eligibility criteria records"""

    queries: List[str] = Field(
        ...,
        description="Distinct queries to search for",
    )
    

output_parser = PydanticToolsParser(tools=[Search])

system = """
You are tasked with a critical role: to dissect a complex, structured query into its component sub-queries. Each component of the query is encapsulated in a JSON dictionary, representing a unique aspect of the information sought. Your objective is to meticulously parse this JSON, isolating each field as a standalone sub-query. These sub-queries are the keys to unlocking detailed, specific information pertinent to each field.

As you embark on this task, remember:
- Treat each JSON field with precision, extracting it as an individual query without altering its essence.
- Your analysis should preserve the integrity of each sub-query, ensuring that the original context and purpose remain intact.
- Enhance each sub-query by contextually expanding it into a complete, meaningful sentence. The aim is to transform each piece of data into a narrative that provides insight into the patient's health condition or medical history.
- Approach this task with the understanding that the fidelity of the sub-queries to their source is paramount. Alterations or misinterpretations could lead to inaccuracies in the information retrieved.

This meticulous separation of the structured query into clear, unmodified sub-queries is fundamental. It enables a tailored search for information, enhancing the relevance and accuracy of the responses generated. Your role in this process is not just to parse data, but to ensure that each piece of information extracted is a faithful reflection of the query's intent, ready to be matched with precise and relevant data points.
"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, api_key=openai_access_key)
structured_llm = llm.with_structured_output(Search)
query_analyzer = {"question": RunnablePassthrough()} | prompt | structured_llm

@chain
async def custom_chain(question):
    response = await query_analyzer.ainvoke(question)
    docs = []
    for query in response.queries:
        new_docs = await retriever.ainvoke(query)
        docs.extend(new_docs)
    # You probably want to think about reranking or deduplicating documents here
    # But that is a separate topic
    return docs

def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Criteria {i+1}:\n\nPage Content: {d.page_content}\nNCT ID: {d.metadata.get('nct_id', 'N/A')}\nCriteria Type: {d.metadata.get('criteria_type', 'N/A')}" for i, d in enumerate(docs)]
        )
    )
    
    
