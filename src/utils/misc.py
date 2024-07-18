def add_custom_entity(doc, entity):
    """
    Add a custom entity to a spaCy document.

    This function takes a spaCy document and a custom entity dictionary and adds the custom entity to the document.
    The function finds the token indices corresponding to the character span of the entity and sets the entity span in the document.

    Parameters:
        doc (spacy.tokens.Doc): The spaCy document to which the entity is added.
        entity (dict): The custom entity dictionary containing the text, start, end, and entity_group.

    Returns:
        spacy.tokens.Doc: The modified spaCy document with the custom entity added.

    Example:
        doc = nlp("The patient has a fever.")
        entity = {"text": "fever", "start": 16, "end": 21, "entity_group": "Symptom"}
        doc = add_custom_entity(doc, entity)
    """
    entity["text"] = re.sub(r'([,.-])\s+', r'\1', entity["text"]) 
    entity_text = entity["text"].lower()
    start_char = entity["start"] 
    end_char = entity["end"] 
    # Find the token indices corresponding to the character span
    start_indices = [i for i, token in enumerate(doc) if (start_char <= token.idx <= end_char) or (entity_text in token.text and token.idx <= start_char)]
    if start_indices:
        # You can choose the first matching window or handle multiple matches
        start_index = start_indices[0]
        start_token = doc[start_index]
        end_index = min(start_index + len(entity_text.split()) - 1, len(doc) - 1)
        end_token = doc[end_index]
        doc.set_ents([Span(doc, start_token.i, end_token.i + 1, entity["entity_group"])])
        
    return doc


def negation_handling(sentence, entity):
    """
    Perform negation handling on a sentence with a given entity.

    This function takes a sentence and an entity dictionary and performs negation handling on the sentence.
    The function uses medSpaCy to identify negation cues and determines if the entity is negated or not.

    Parameters:
        sentence (str): The sentence in which the entity is present.
        entity (dict): The entity dictionary containing the text, start, and end.

    Returns:
        dict: The modified entity dictionary with the "is_negated" field indicating if the entity is negated or not.

    Example:
        sentence = "The patient does not have a fever."
        entity = {"text": "fever", "start": 23, "end": 28}
        entity = negation_handling(sentence, entity)
    """
    nlp = spacy.load("en_core_web_sm", disable={"ner"})
    doc = nlp(sentence.lower())
    nlp = medspacy.load(nlp)
    nlp.disable_pipe('medspacy_target_matcher')
    nlp.disable_pipe('medspacy_pyrush')
    doc = nlp(add_custom_entity(doc, entity))
    for e in doc.ents:
        rs = str(e._.is_negated)
        if rs:
            if rs == "True": 
                entity["is_negated"] = "yes"
            elif rs == 'False':
                entity["is_negated"] = "no"
        else:
            entity["is_negated"] = "no"
    return entity