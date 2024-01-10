# preprocessing_utils.py

"""
Description: This script contains functions for pre-processing clinical trials eligibility criteria texts.  
The functions serve to split the raw unstructured text into clean and structured sentences to be processed by a more advanced downstream NLP analysis
"""

import numpy as np
import re
import itertools
from itertools import islice
import pandas as pd
import json 
import xml.etree.ElementTree as ET
import os
import re
import logging
import nltk

# Configure logging
# logging.basicConfig(filename='../logs/app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')


def flatten_list_of_lists(list_of_lists):
    """
    Flatten a list of lists into a single list.

    Parameters:
        list_of_lists (list): The list of lists to be flattened.

    Returns:
        list: A flattened list.
    """
    return [item for sublist in list_of_lists for item in sublist]

def load_regex_patterns(file_path):
    """
    Load regular expression patterns from a JSON file.

    This function reads a JSON file containing regular expression patterns and extracts the patterns
    into a dictionary. The JSON file should have a specific structure with the following elements:
    {
        "patterns": {
            "pattern_name1": {
                "regex": "pattern_expression1"
            },
            "pattern_name2": {
                "regex": "pattern_expression2"
            },
            ...
        }
    }
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
        patterns = {key: value['regex'] for key, value in data['patterns'].items()}
    return patterns


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

def read_xml_file(file_path):
    try:
        with open(file_path, 'r') as xml_file:
            return xml_file.read()
    except IOError as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return None

def parse_xml_content(xml_content):
    try:
        tree = ET.ElementTree(ET.fromstring(xml_content))
        return tree.getroot()
    except ET.ParseError as e:
        logging.error(f"Error parsing XML content: {e}")
        return None

def extract_eligibility_criteria(trial_id):
    """
    Extract the eligibility criteria text for a clinical trial with the given trial ID.
    """
    xml_file_path = os.path.join('..', 'data', 'trials_xmls', f'{trial_id}.xml')

    if os.path.exists(xml_file_path):
        xml_content = read_xml_file(xml_file_path)
        if xml_content is None:
            return None

        root = parse_xml_content(xml_content)
        if root is None:
            return None

        eligibility_criteria_textblock = root.find(".//eligibility/criteria/textblock")
        if eligibility_criteria_textblock is not None:
            return eligibility_criteria_textblock.text.strip()
        else:
            logging.warning(f"Eligibility criteria textblock not found for trial ID {trial_id}.")
            return None

    logging.warning(f"XML file for trial ID {trial_id} not found.")
    return None

# def extract_eligibility_criteria(trial_id):
#     """
#     Extract the eligibility criteria text for a clinical trial with the given trial ID.

#     This function attempts to locate and extract the eligibility criteria text for a clinical trial
#     specified by its trial ID. The function reads an XML file named '{trial_id}.xml' which is expected
#     to contain information for the clinical trial. It searches for the eligibility criteria textblock within
#     the XML and extracts the corresponding text.

#     Parameters:
#         trial_id (str): The unique identifier of the clinical trial.

#     Returns:
#         str or None: The extracted eligibility criteria text for the specified trial if found,
#                     otherwise None.
#     """
#     xml_file_path = f'../data/trials_xmls/{trial_id}.xml'

#     if os.path.exists(xml_file_path):
#         with open(xml_file_path, 'r') as xml_file:
#             xml_content = xml_file.read()
#         try:
#             tree = ET.ElementTree(ET.fromstring(xml_content))
#             root = tree.getroot()
#         except ET.ParseError as e:
#             print(f"Error parsing XML for trial {trial_id}: {e}")
#             return None
#         # Find the Eligibility Criteria TextBlock section within the XML
#         eligibility_criteria_textblock = root.find(".//eligibility/criteria/textblock")

#         if eligibility_criteria_textblock is not None:
#             # Extract the text from the Eligibility Criteria TextBlock section
#             eligibility_criteria_text = eligibility_criteria_textblock.text
#             return eligibility_criteria_text.strip()
#     else:
#     # If the trial ID is not found or the eligibility criteria textblock is missing, return None
#         return None

def split_by_leading_char_from_regex_patterns(line, regex_patterns, exceptions_path = "../data/exception_regex_patterns.json"):
    """
    Split a line of text into sentences using leading characters defined by regex patterns.

    This function takes a line of text and splits it into sentences based on leading characters defined by regular expression (regex) patterns.
    It is useful for scenarios where sentences in the text are indicated by specific patterns at the beginning of a word.

    The function iterates through the words in the input line and identifies the sentences by matching each word against the provided
    regex patterns. If a word matches any of the regex patterns, it is considered the start of a new sentence. The function then appends
    the completed sentence to the list of sentences. The process continues until all words are processed.

    An exception pattern can also be provided to prevent sentence splitting based on certain word patterns. If a word matches any of
    the exception patterns, it is included in the current sentence rather than being considered as the start of a new sentence.

    Parameters:
        line (str): The input line of text to be split into sentences.
        regex_patterns (list): A list of regular expression patterns. Words matching any of these patterns are considered the start of new sentences.
        exception_patterns (str): Optional. The file path to an exceptions file containing regex patterns. 
        Words matching any of these exception patterns are included in the current sentence rather than starting new sentences.

    Returns:
        list: A list of sentences extracted from the input line.
    """
    sentences = []
    sentence = ""
    # Replace parentheses with braces in the line
    line = replace_parentheses_with_braces(line)
    words = line.split()  # Split the line into words
    for i, word in enumerate(words):
        if i < len(words) - 3:
            is_match = any([
                re.match(pattern, word) for pattern in list(load_regex_patterns(exceptions_path).values())
            ])
            if is_match:
                sentence += word + " "
            else:
                for pattern in regex_patterns:
                    if re.search(pattern, word):
                        if sentence != "":
                            sentences.append(sentence.strip())
                            sentence = ""
                        break
                sentence += word + " "
        else:
            sentence += word + " "
    if sentence != "":
        sentences.append(sentence.strip())
    return sentences


def is_header(line, next_line, regex_patterns):
    """
    Determine if a line is a header based on specific criteria.

    This function takes two lines of text and a list of regular expression (regex) patterns, and it determines if the first line is a header
    based on specific criteria. It is designed to identify headers in text documents.

    The function considers various conditions to classify a line as a header. It checks if the line ends with a colon and matches any of
    the provided regex patterns. It also checks if the line starts with an uppercase letter and ends with a colon, or if it starts with an
    uppercase letter and doesn't end with a colon but the next line starts with a regex pattern or has a higher indentation level.

    Parameters:
        line (str): The first line of text to be checked for being a header.
        next_line (str): The next line of text following the first line.
        regex_patterns (list): A list of regular expression patterns to match against the line.

    Returns:
        bool: True if the line is considered a header, False otherwise.

    Example:
        line = "Introduction:"
        next_line = "This is the introduction to the topic."
        regex_patterns = [r"Chapter \d+", r"Section \d+"]
        is_header(line, next_line, regex_patterns)
        # Output: True
    """
    line_indent = len(line) - len(line.lstrip())
    next_line_indent = len(next_line) - len(next_line.lstrip())

    # Check if the line ends with a colon and matches any regex pattern
    if any(re.match(pattern, line) for pattern in regex_patterns) and line.rstrip().endswith(":"):
        return True

    # Check if the line starts with an uppercase letter and ends with a colon
    if line[0].isupper() and line.rstrip().endswith(":"):
        return True

    # Check if the line doesn't start with an uppercase letter, but ends with a colon
    if not line[0].isupper() and line.rstrip().endswith(":"):
        return True

    # Check if the line starts with an uppercase letter and doesn't end with a colon,
    # and either the next line starts with a regex pattern or has a higher indentation level
    if line[0].isupper() and not line.rstrip().endswith(":") and (any(re.match(pattern, next_line) for pattern in regex_patterns) or line_indent < next_line_indent):
        return True

    # Check if the line doesn't end with a colon and doesn't start with an uppercase letter,
    # and the next line has a higher indentation level
    if not line.rstrip().endswith(":") and not line[0].isupper() and line_indent < next_line_indent:
        return True
    # Check if the line doesn't end with a colon, doesn't start with an uppercase letter,
    # doesn't match any regex pattern, and either the next line starts with a regex pattern
    # or has a higher indentation level
    if not line.rstrip().endswith(":") and not any(re.match(pattern, line) for pattern in regex_patterns) and not (re.match(r"^[A-Za-z]", line) or line[0].isupper()) and (any(re.match(pattern, next_line) for pattern in regex_patterns) or line_indent < next_line_indent):
        return True

    return False  # If none of the conditions are met, it's not a header


def is_false_header(line, prev_line, next_line):
    """
    Determine if a line is a false header based on specific criteria.

    This function takes three lines of text and determines if the first line is a false header based on specific criteria.
    It is designed to identify lines that might appear as headers but are not actual headers in text documents.

    The function considers various conditions to classify a line as a false header. It checks if the line ends with a colon
    but starts with a lowercase letter or a number. It also checks if the line directly before the header line ends with a comma.
    Additionally, it checks if the indentation level of the header line is greater than the line after it.

    Parameters:
        line (str): The line of text to be checked for being a false header.
        prev_line (str): The line of text directly before the line being checked.
        next_line (str): The line of text following the line being checked.

    Returns:
        bool: True if the line is considered a false header, False otherwise.

    Example:
        line = "introduction:"
        prev_line = "This is the introduction to the topic,"
        next_line = "and it explains the main concepts."
        is_false_header(line, prev_line, next_line)
        # Output: True
    """
    line_indent = len(line) - len(line.lstrip())
    next_line_indent = len(next_line) - len(next_line.lstrip())

    # Condition 1: Header line ends with a colon but starts with a lowercase letter or a number
    if line.rstrip().endswith(":") and (line[0].islower() or line[0].isdigit()):
        return True

    # Condition 2: The line directly before the header line ends with a comma
    if prev_line.rstrip().endswith(","):
        return True

    # Condition 3: Header line has a greater indentation level than the line after it
    if line_indent > next_line_indent:
        return True

    return False

# Define constants for line types
LINE_TYPE_REGULAR = 0
LINE_TYPE_HEADER = 1
LINE_TYPE_FALSE_HEADER = 2

def split_on_carriage_returns(text, regex_patterns):
    """
    Split a text into lines separated by double carriage returns (i.e. \n\n)

    This function takes a text and a list of regular expression (regex) patterns. It splits the text into lines using double carriage returns.
    For each line, it identifies the type based on certain conditions, including whether it is a header or a continuation of the previous line.

    Parameters:
        text (str): The input text to be split into lines and identified.
        regex_patterns (list): A list of regular expression patterns to match against the lines.

    Returns:
        list: A list of tuples, where each tuple contains a line and its corresponding type:
            - Type 0: Regular line
            - Type 1: Header line
            - Type 2: False header line (appears as a header but is not an actual header)

    Example:
        text = "Introduction:\n\nThis is the introduction to the topic.\n\n"
        regex_patterns = [r"Chapter \d+", r"Section \d+"]
        split_on_carriage_returns(text, regex_patterns)
        # Output: [(Introduction:, 1), (This is the introduction to the topic., 0)]
    """
    lines = re.split(r'\n\n+', re.sub(r':\n', ':\n\n', text)) # Split the text into lines using double carriage returns
    result = []
    current_line = ""
    line_type = LINE_TYPE_REGULAR

    for i, line in enumerate(lines):
        current_line += line

        if i == len(lines) - 1:
            result.append((current_line, line_type))
            break

        if is_header(lines[i].lstrip(), lines[i + 1].lstrip(), regex_patterns):
            line_type = LINE_TYPE_HEADER

        if (not any(re.search(pattern, lines[i + 1].lstrip()) for pattern in regex_patterns) and lines[i].rstrip().endswith((",", ";"))) and not line_starts_with_capitalized_alphanumeric(lines[i+1].lstrip()):
            current_line += " " + lines[i + 1]
            i += 1

        elif i < len(lines) - 2 and is_header(lines[i + 1].lstrip(), lines[i + 2].lstrip(), regex_patterns):
            if not is_false_header(lines[i + 1], lines[i], lines[i + 2]):
                i += 1
            elif is_false_header(lines[i + 1], lines[i], lines[i + 2]):
                current_line += " " + lines[i+1]
                line_type = LINE_TYPE_FALSE_HEADER
                i += 1

        current_line = re.sub(r'\s+', ' ', current_line)
        result.append((current_line, line_type))
        current_line = ""
        line_type = LINE_TYPE_REGULAR

    return result

# def split_on_carriage_returns(text, regex_patterns):
#     """
#     Split a text into lines separated by double carriage returns (i.e. \n\n)

#     This function takes a text and a list of regular expression (regex) patterns. It splits the text into lines using double carriage returns.
#     For each line, it identifies the type based on certain conditions, including whether it is a header or a continuation of the previous line.

#     Parameters:
#         text (str): The input text to be split into lines and identified.
#         regex_patterns (list): A list of regular expression patterns to match against the lines.

#     Returns:
#         list: A list of tuples, where each tuple contains a line and its corresponding type:
#             - Type 0: Regular line
#             - Type 1: Header line
#             - Type 2: False header line (appears as a header but is not an actual header)

#     Example:
#         text = "Introduction:\n\nThis is the introduction to the topic.\n\n"
#         regex_patterns = [r"Chapter \d+", r"Section \d+"]
#         split_on_carriage_returns(text, regex_patterns)
#         # Output: [(Introduction:, 1), (This is the introduction to the topic., 0)]
#     """
#     lines = re.split(r'\n\n+', re.sub(r':\n', ':\n\n', text)) # Split the text into lines using double carriage returns
#     result = []
#     current_line = ""
#     i = 0
#     while i < len(lines) :
#         line = lines[i]
#         current_line += line
#         line_type = 0          
#         if i == len(lines) - 1 :
#             i += 1
#             result.append((current_line, line_type))
#             break
#         if is_header(lines[i].lstrip(), lines[i + 1].lstrip(), regex_patterns) :
#             line_type = 1   
#         if (not any(re.search(pattern, lines[i + 1].lstrip()) for pattern in regex_patterns) and lines[i].rstrip().endswith((",", ";"))) and not line_starts_with_capitalized_alphanumeric(lines[i+1].lstrip()) :
#             current_line += " " + lines[i + 1] 
#             i += 2
#         elif i < len(lines) - 2 and is_header(lines[i + 1].lstrip(), lines[i + 2].lstrip(), regex_patterns) :
#             if not is_false_header(lines[i + 1], lines[i], lines[i + 2]):
#                 i += 1
#             elif  is_false_header(lines[i + 1], lines[i], lines[i + 2]):
#                 current_line += " " + lines[i+1]
#                 line_type = 2
#                 i += 2   
#         else:
#             i += 1
#         current_line = re.sub(r'\s+', ' ', current_line)
#         result.append((current_line, line_type))
#         current_line = ""
#         line_type = ""
#     return result
    
def split_to_sentences(text, regex_patterns):
    """
    Split a text into sentences based on specific criteria.

    This function takes a text and a list of regular expression (regex) patterns. It first splits the text into lines and identifies the type
    of each line using the 'split_on_carriage_returns' function. Then, for each line, it further splits it into sentences using the
    'split_by_leading_char_from_regex_patterns' function based on specific criteria. The resulting sentences are
    filtered to include only those with more than 1 word.

    Parameters:
        text (str): The input text to be split into sentences.
        regex_patterns (list): A list of regular expression patterns to match against the lines.

    Returns:
        list: A list of sentences extracted from the text.

    Note:
    The `split_on_carriage_returns` and `split_by_leading_char_from_regex_patterns` functions must be defined and imported
    to use this function.

    See Also:
    split_on_carriage_returns
    split_by_leading_char_from_regex_patterns
    """
    lines = split_on_carriage_returns(text, regex_patterns)
    cleaned_lines = []
    for i in range(len(lines)):
        line = lines[i][0].strip()
        if i < len(lines) - 1:
            next_line = lines[i+1][0].strip()
            if not next_line or next_line.startswith('-') or re.search(r'\s{2,}', next_line) or re.search(r'^\d+\s*\.', next_line):
                line += ' '
        line = re.sub(r"\n", " ", line)
        line = split_by_leading_char_from_regex_patterns(line, regex_patterns)
        line = [string for string in line if len(string.split()) > 1]
        cleaned_lines.append(line)
    flat_list = [item for sublist in cleaned_lines for item in sublist]
    return flat_list


def drop_leading_character(sentence, regex_patterns):
    """
    Drop leading characters from a sentence based on regex patterns.

    This function takes a sentence and a list of regular expression (regex) patterns. It iterates over the regex patterns, and for each
    pattern, it drops the leading character from the sentence if there is a match. The loop continues until no more matches are found
    for any of the patterns. The resulting sentence is then stripped of leading and trailing whitespaces.

    Parameters:
        sentence (str): The input sentence from which leading characters will be dropped.
        regex_patterns (list): A list of regular expression patterns to match against the leading characters.

    Returns:
        str: The sentence with leading characters dropped.

    Example:
        sentence = "A. This is a sample sentence."
        regex_patterns = [r"^[A-Z]\.", r"^\d+\."]
        drop_leading_character(sentence, regex_patterns)
        # Output: "This is a sample sentence."
    """
    for pattern in regex_patterns:
        while True:
            match = re.match(pattern, sentence)
            if match:
                # Drop the leading character by substituting it with an empty string,
                # but only replace the first occurrence
                sentence = re.sub(pattern, '', sentence, count=1).strip()
            else:
                # If no more matches found, exit the loop
                break
    return sentence.strip()


def extract_criteria_sections_headers(lines):
    """
    Extract criteria sub-sections headers from a list of lines.

    This function takes a list of lines, originally from the clinical trial texts, as input and extracts headers for inclusion and exclusion criteria sub-sections from the list. 
    It uses explicit regular expression (regex) patterns to identify various writing styles of group-specific criteria headers. The extracted headers are
    returned as a dictionary with each header as a key and the list of line indices where the header occurs as the value.

    Parameters:
        lines (list): A list of strings representing the lines of text.

    Returns:
        dict: A dictionary containing the extracted criteria sections headers.

    Example:
        lines = [
            "Inclusion Criteria - Group A:",
            "Key Exclusion Criteria for Subjects with Diabetes:",
            "Eligibility Requirements for Patients",
            "Exclusion: Patients with Allergies",
            "Patients - Inclusion Criteria:"
        ]
        extract_criteria_sections_headers(lines)
        # Output: {
        #    "Inclusion Criteria - Group A": [0],
        #    "Key Exclusion Criteria for Subjects with Diabetes": [1],
        #    "Eligibility Requirements for Patients": [2],
        #    "Exclusion: Patients with Allergies": [3],
        #    "Patients - Inclusion Criteria": [4]
        # }

    Note:
    The function uses predefined regex patterns to identify various writing styles for criteria section headers. The patterns are designed
    to match common variations of headers in clinical trial eligibility criteria.
    """
    criteria_sections = {}
    # Define explicit patterns for different writing styles of group-specific criteria headers
    patterns = [
    r"^(?:Inclusion|Exclusion|Eligibility|Selection)\s(?:Criteria|Requirements?)?\s(?:for|in)?\s(?:Patients|Subjects|Population|Cohort|Group|Arm)\s?(?:with|without|who|where|having)?\s?[\w\d\s-]*[:\-]?",
    r"^(?:Key\s)?(?:Inclusion|Exclusion|Eligibility|Selection)(?:\s(?:Criteria|Requirements))?(?:\s?[-+:]|\sfor)?(?:\s[\w\s+-]+)?(?:\([\w\s]+\))?\s?[-+:]?\s?[\w\s]+$",
    r"^(?:Key\s)?(?:Inclusion|Exclusion|Eligibility|Selection)(?:\s(?:Criteria|Requirements?))(?:\s(?:for|in))?(?:\s(?:Patients|Subjects|Population|Cohort|Group|Arm))?(?:\s(?:with|without|who|where|having))?\s?(?:\([\w\s]+\))?\s?[\w\s+-]*[:\-]?",
    r"^(?:[\w\d\s-]+)\s*-\s*(?:Inclusion|Exclusion|Eligibility|Selection)\s(?:Criteria|Requirements?)?$",
    r"^(?:[\w\s]+?)\s(?:group|patients|population|arm|subjects|cohort)\s(?:inclusion|exclusion|eligibility|selection|criteria)(?:\s?:|-)?",
    r"^\b(?:\w+\s\w+|\w+)?\s(?:Inclusion|Exclusion|Eligibility|Selection)\s(?:Criteria|Requirements)\b",
    ] 
    for i, line in enumerate(lines):       
        if ":" in line.rstrip():
            line = line.split(":")[0].strip()
        if len(line.split()) <= 10:
            if any(re.search(pattern, line, re.IGNORECASE) for pattern in patterns) : 
                line = line + " "
                header = line.strip() 
                if header not in criteria_sections:
                    criteria_sections[header] = [i]
                else:
                    criteria_sections[header].extend([i])
    return criteria_sections


def extract_seperate_inclusion_exclusion(text, regex_patterns):
    """
    Function to extract preprocessed inclusion and exclusion criteria from clinical trials eligibility criteria text.

    This function takes raw text and extracts Inclusion Criteria, Exclusion Criteria, and also the Original Eligibility Criteria. 
    It uses the provided regex patterns to split the text into sentences and identify criteria sub-sections headers.

    Parameters:
        text (str): The preprocessed text containing eligibility criteria.
        regex_patterns (list): A list of regular expression patterns used to split the text into sentences.

    Returns:
        dict: A dictionary containing the extracted Inclusion Criteria, Exclusion Criteria, and Original Eligibility Criteria.

    Note:
    The function uses the regex patterns to split the text into sentences and identify headers for Inclusion and Exclusion Criteria
    sections. It then processes the sentences to group them into corresponding criteria sections.

    See Also:
    split_to_sentences
    extract_criteria_sections_headers
    """
    criteria = {
        "Inclusion Criteria": {},
        "Exclusion Criteria": {},
        "Original Eligibility Criteria": text
    }
    
    lines = split_to_sentences(text, regex_patterns)
    subsection_indices = extract_criteria_sections_headers(lines)
    inclusion_pattern = r"(?<!\S)(?:inclusion|eligibility|selection|included|are eligible)(?!\S|$)"
    exclusion_pattern = r"(?<!\S)(?:exclusion|non-inclusion|excluded|not eligible|non-selection)(?!\S|$)"
    inclusion_indices = np.sort(list(itertools.chain(*[value for _, (key,value) in enumerate(subsection_indices.items()) if re.search(inclusion_pattern, key, re.IGNORECASE)])))
    exclusion_indices = np.sort(list(itertools.chain(*[value for _, (key,value) in enumerate(subsection_indices.items()) if re.search(exclusion_pattern, key, re.IGNORECASE)])))
    num_inclusion = len(inclusion_indices)
    num_exclusion = len(exclusion_indices)
    if num_inclusion >= 1 and num_exclusion == 0:
            for i in range(num_inclusion):
                inclusion_start_index = inclusion_indices[i]
                inclusion_end_index = inclusion_indices[i+1] if i < num_inclusion - 1 else None
                inclusion_criteria = lines[inclusion_start_index:inclusion_end_index]
                criteria["Inclusion Criteria"][f"{lines[inclusion_indices[i]].strip()}"] = inclusion_criteria
                    
    elif num_inclusion == 0 and num_exclusion >= 1:
            for i in range(num_exclusion):
                exclusion_start_index = exclusion_indices[i]
                exclusion_end_index = exclusion_indices[i + 1] if i < num_exclusion - 1 else None
                exclusion_criteria = lines[exclusion_start_index:exclusion_end_index]
                criteria["Exclusion Criteria"][f"{lines[exclusion_indices[i]].strip()}"] = exclusion_criteria
                
    elif num_inclusion == 1 and num_exclusion == 1:
        inclusion_start_index = inclusion_indices[0]
        exclusion_start_index = exclusion_indices[0] if num_exclusion > 0 else None
        inclusion_criteria = lines[inclusion_start_index:exclusion_start_index] 
        criteria["Inclusion Criteria"] = inclusion_criteria
        exclusion_criteria = lines[exclusion_start_index:] if num_exclusion > 0 else None
        criteria["Exclusion Criteria"] = exclusion_criteria
        
    else:
        for i in range(num_inclusion):
            inclusion_start_index = inclusion_indices[i]
            if i < num_inclusion - 1 and any(inclusion_indices[i+1] > x for x in exclusion_indices):
                inclusion_end_index = exclusion_indices[np.argwhere(exclusion_indices < inclusion_indices[i+1])].flatten()[0]  
            elif i == num_inclusion - 1 and any(inclusion_indices[i] < x for x in exclusion_indices): 
                inclusion_end_index = exclusion_indices[np.argwhere(exclusion_indices > inclusion_indices[i])].flatten()[0] 
            elif i < num_inclusion - 1 and not any(inclusion_indices[i+1] > x for x in exclusion_indices):
                inclusion_end_index = inclusion_indices[i+1]
            inclusion_criteria = lines[inclusion_start_index:inclusion_end_index]
            criteria["Inclusion Criteria"][f"{lines[inclusion_indices[i]].strip()}"] = inclusion_criteria

        for i in range(num_exclusion):
            exclusion_start_index = exclusion_indices[i]
            if  any(exclusion_indices[i] < x for x in inclusion_indices) and num_exclusion >= 1:
                exclusion_end_index = inclusion_indices[np.argwhere(inclusion_indices > exclusion_indices[i])].flatten()[0] 
            elif any(exclusion_indices[i] < x for x in inclusion_indices) and num_exclusion > 1 and exclusion_indices[i + 1] < inclusion_indices[np.argwhere(inclusion_indices > exclusion_indices[i])].flatten()[0]:
                exclusion_end_index = exclusion_indices[i + 1]
            elif all(exclusion_indices[i] > x for x in inclusion_indices) and num_exclusion > 1 and i < num_exclusion - 1 :
                exclusion_end_index = exclusion_indices[i + 1]
            elif all(exclusion_indices[i] > x for x in inclusion_indices) and num_exclusion >= 1 and i == num_exclusion - 1:
                exclusion_end_index= None
            exclusion_criteria = lines[exclusion_start_index:exclusion_end_index]
            criteria["Exclusion Criteria"][f"{lines[exclusion_indices[i]].strip()}"] = exclusion_criteria

    return criteria


def eic_text_preprocessing(_ids, regex_path = "../data/regex_patterns.json", output_path = "../data/preprocessed_data/clinical_trials/"):
    """
    Main preprocessing function for eligibility criteria text from a list of clinical trial IDs.

    This function takes a list of clinical trial IDs (_ids) and preprocesses the eligibility criteria text
    for each trial. It uses the provided regex patterns to extract Inclusion Criteria and Exclusion Criteria from the text.

    Parameters:
        _ids (list): A list of clinical trial IDs for which eligibility criteria text will be preprocessed.
        regex_patterns (dict): A dictionary containing regular expression patterns used for preprocessing.

    Returns:
        pandas.DataFrame: A DataFrame containing the preprocessed eligibility criteria text with columns
        "sentence," "criteria," "sub_criteria," and "_id."

    Note:
    The function calls extract_eligibility_criteria to obtain the eligibility criteria text for each trial.
    It then uses the extract_seperate_inclusion_exclusion function to preprocess the eligibility criteria text for each trial,
    extracting Inclusion Criteria, Exclusion Criteria, and Original Eligibility Criteria. The results are concatenated
    into a final DataFrame.

    See Also:
    extract_eligibility_criteria
    extract_seperate_inclusion_exclusion
    drop_leading_character
    """
    regex_list = list(load_regex_patterns(regex_path).values())
    texts  = []
    trial_id = []
    for _, nid in enumerate(_ids):
        eic_text = extract_eligibility_criteria(nid)
        if eic_text:
            texts.append(extract_seperate_inclusion_exclusion(eic_text, regex_list))
            trial_id.append(nid)
        else:
            continue
    to_concat = []
    for index, item in enumerate(texts):
        iterator = islice(item.items(), 2)
        _id = trial_id[index]  # Get the NCT ID for the current item
        for key, value in iterator:
            if isinstance(value, dict):  # Check if the value is a dictionary
                for sub_key, sub_value in value.items():
                    df = pd.DataFrame(sub_value, columns=["sentence"])
                    df["criteria"] = key
                    df["sub_criteria"] = sub_key
                    df["id"] = _id
                    to_concat.append(df)
            else:
                df = pd.DataFrame(value, columns=["sentence"])
                df["criteria"] = key
                df["sub_criteria"] = key  # Use key as sub-criteria when value is not a dictionary
                df["id"] = _id
                to_concat.append(df)
    if to_concat:
        final_df = pd.concat(to_concat)
        final_df['sentence'] = final_df['sentence'].apply(drop_leading_character, regex_patterns=regex_list)
        final_df.to_csv(output_path + "%s_preprocessed.csv"%_ids[0])
        return final_df
    else:
        return None
    
    
    