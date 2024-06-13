import torch
from transformers import pipeline

def generate_sentence_from_keyword(keyword, pipe):
    # Define the messages with the improved system message content
    messages = [
        {
            "role": "system",
            "content": (
                "You are a clinical assistant responsible for creating concise sentences from keywords. "
                "These keywords are related to a cancer patient's clinical conditions and genetic markers. "
                "Your task is to generate a meaningful sentence that includes a keyword in a clear and relevant medical context. "
                "The sentence should be concise and accurately describes the patient's condition or genetic status and suitable for querying a database of clinical trials eligibility criteria. "
                "Do not add extra information that is not found in the keyword."
                "If you are not familiar with the keyword, keep it as a single word. Do not try to add any additional words."
                "Example: if keyword is 'KRAS', a suitable sentence could be 'The patient has a mutation in the KRAS gene.'"
                "Another example: if keyword is 'Adjuvant Chemotherapy', a suitable sentence could be 'The patient has undergone adjuvant chemotherapy.'"
                "Another example: if keyword is 'BRCA1', a suitable sentence could be 'The patient carries a mutation in the BRCA1 gene.'"
            ),
        },
        {"role": "user", "content": keyword},
    ]

    # Apply the chat template using the tokenizer's method
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Generate the output
    outputs = pipe(
        prompt, 
        max_new_tokens=30,  # Reduce token limit for conciseness
        do_sample=True, 
        temperature=0.25, 
        top_k=50, 
        top_p=0.95
    )

    # Extract and return the generated text, removing the prompt
    generated_text = outputs[0]["generated_text"]
    generated_sentence = generated_text.split("</s>")[-1].strip()
    return generated_sentence

def generate_sentences_from_keywords(keywords):
    # Initialize the text generation pipeline
    pipe = pipeline(
        "text-generation", 
        model="HuggingFaceH4/zephyr-7b-alpha", 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )

    sentences = []
    for keyword in keywords:
        sentence = generate_sentence_from_keyword(keyword, pipe)
        sentences.append(sentence)
    return sentences

# Example usage
if __name__ == "__main__":
    keywords = ["KRAS", "BRCA1", "TP53", "Adjuvant Chemotherapy", "Immunotherapy"]  # Replace with the actual keywords as needed
    expanded_sentences = generate_sentences_from_keywords(keywords)
    for keyword, sentence in zip(keywords, expanded_sentences):
        print(f"Keyword: {keyword}, Expanded Sentence: {sentence}")
