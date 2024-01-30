from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import re

tokenizer = AutoTokenizer.from_pretrained("bvanaken/clinical-assertion-negation-bert")
model = AutoModelForSequenceClassification.from_pretrained("bvanaken/clinical-assertion-negation-bert")
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

def is_entity_negated(sentence, entity):
    # Surround the entity with [entity] on both sides
    sentence_with_entity = sentence.replace(entity, f"[entity]{entity}[entity]")

    # Classify the modified sentence to check for negation
    classification = classifier(sentence_with_entity)[0]
    is_negated = classification['label'] == 'ABSENT'

    return is_negated

# Example usage:
sentence = "The patient recovered during the night and now denies any shortness of breath."
entity = "shortness of breath"
result = is_entity_negated(sentence, entity)
print(result)

# [{'label': 'ABSENT', 'score': 0.9842607378959656}]