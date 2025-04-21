"""
Description: This script contains functions for pre-processing clinical trials eligibility criteria texts.  
The functions serve to split the raw unstructured text into clean and structured sentences to be processed by a more advanced downstream NLP analysis.
"""

import os
import re
import json
import logging
import itertools
import pandas as pd
import xml.etree.ElementTree as ET
import csv

def load_regex_patterns(file_path):
    """
    Load regular expression patterns from a JSON file.

    Parameters:
        file_path (str): Path to the JSON file containing regex patterns.

    Returns:
        dict: A dictionary with pattern names as keys and regex patterns as values.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
        patterns = {key: value['regex'] for key, value in data['patterns'].items()}
    return patterns


def split_on_leading_markers(lines):
    """
    Attempt to split lines on various common list markers:
    - Bullets (•)
    - Dashes (-) at the start of items
    - Numeric or alphabetical lists (e.g., "1)", "a)")
    Adjust the patterns to fit your data.
    """
    new_lines = []
    for line in lines:
        # First split on bullet points
        bullet_parts = [p.strip() for p in re.split(r'•', line) if p.strip()]
        temp_lines = []
        for part in bullet_parts:
            # Split on leading dashes
            # This will split lines like "- Something" or multiple dashes in a single line.
            dash_parts = [dp.strip() for dp in re.split(r'(?<!^)(?=-\s)', part) if dp.strip()]
            for dp in dash_parts:
                # Split on asterisks (*) instead of numeric or alphabetic patterns
                # Example pattern: split on items that start with an asterisk followed by a space
                final_parts = re.split(r'(?<!^)(?=\*)', dp)
                final_parts = [f.strip() for f in final_parts if f.strip()]
                temp_lines.extend(final_parts)
        if not temp_lines:
            # If no splitting occurred at all, just add the original line
            temp_lines.append(line.strip())
        new_lines.extend(temp_lines)
    return new_lines


def replace_parentheses_with_braces(text):
    """
    Replace parentheses and brackets with curly braces in the given text.

    Parameters:
        text (str): The input text.

    Returns:
        str: The modified text with parentheses and brackets replaced by curly braces.
    """
    stack = []
    result = ""
    for char in text:
        if char in '([':
            stack.append(char)
            result += "{"
        elif char in ')]':
            if stack:
                stack.pop()
                result += "}"
            else:
                result += char
        else:
            result += char
    return result


def replace_braces_with_parentheses(text):
    """
    Replace curly braces with parentheses in the given text.

    Parameters:
        text (str): The input text containing curly braces.

    Returns:
        str: The text with curly braces replaced by parentheses.
    """
    return text.replace('{', '(').replace('}', ')')


def line_starts_with_capitalized_alphanumeric(line):
    """
    Check if the line starts with a capitalized alphanumeric character.

    Parameters:
        line (str): The input line.

    Returns:
        bool: True if the line starts with a capitalized alphanumeric character, False otherwise.
    """
    words = line.strip().split()
    if words:
        first_word = words[0]
        return first_word[0].isalpha() and first_word[0].isupper()
    return False


def read_xml_file(file_path):
    """
    Read the content of an XML file.

    Parameters:
        file_path (str): Path to the XML file.

    Returns:
        str or None: The content of the XML file or None if an error occurs.
    """
    try:
        with open(file_path, 'r') as xml_file:
            return xml_file.read()
    except IOError as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return None


def parse_xml_content(xml_content):
    """
    Parse XML content and return the root element.

    Parameters:
        xml_content (str): The XML content as a string.

    Returns:
        xml.etree.ElementTree.Element or None: The root element or None if parsing fails.
    """
    try:
        root = ET.fromstring(xml_content)
        return root
    except ET.ParseError as e:
        logging.error(f"Error parsing XML content: {e}")
        return None


def extract_eligibility_criteria(trial_id):
    """
    Extract the eligibility criteria text for a clinical trial with the given trial ID.

    Parameters:
        trial_id (str): The clinical trial ID.

    Returns:
        str or None: The eligibility criteria text or None if not found.
    """
    xml_file_path = os.path.join('..', '..', 'data', 'trials_xmls', f'{trial_id}.xml')

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


def split_by_leading_char_from_regex_patterns(line, regex_patterns, exceptions_patterns=None):
    """
    Split a line of text into sentences using leading characters defined by regex patterns.

    Parameters:
        line (str): The input line of text.
        regex_patterns (list): A list of regex patterns to split the text.
        exceptions_patterns (list): A list of regex patterns to ignore during splitting.

    Returns:
        list: A list of sentences extracted from the input line.
    """
    if exceptions_patterns is None:
        exceptions_patterns = []

    sentences = []
    combined_pattern = '|'.join(f'({pattern})' for pattern in regex_patterns)
    exception_pattern = '|'.join(f'({pattern})' for pattern in exceptions_patterns)
    last_split = 0

    for match in re.finditer(combined_pattern, line):
        start, end = match.span()

        # Check for exceptions
        if exception_pattern and re.search(exception_pattern, line[start:end]):
            continue

        # Add the sentence up to this match
        if last_split != start:
            sentences.append(line[last_split:start].strip())
        
        last_split = start

    # Add the last segment
    if last_split < len(line):
        sentences.append(line[last_split:].strip())

    return sentences


def is_header(line, next_line, regex_patterns):
    """
    Determine if a line is a header based on specific criteria.

    Parameters:
        line (str): The current line.
        next_line (str): The next line.
        regex_patterns (list): A list of regex patterns.

    Returns:
        bool: True if the line is considered a header, False otherwise.
    """
    if not line:
        return False

    line_indent = len(line) - len(line.lstrip())
    next_line_indent = len(next_line) - len(next_line.lstrip())

    # Check if the line ends with a colon and matches any regex pattern
    if any(re.match(pattern, line) for pattern in regex_patterns) and line.rstrip().endswith(":"):
        return True

    # Check if the line starts with an uppercase letter and ends with a colon
    if line[0].isupper() and line.rstrip().endswith(":"):
        return True

    # Check if the line starts with an uppercase letter and doesn't end with a colon,
    # and either the next line starts with a regex pattern or has a higher indentation level
    if line[0].isupper() and not line.rstrip().endswith(":") and (
        any(re.match(pattern, next_line) for pattern in regex_patterns) or line_indent < next_line_indent):
        return True

    return False


def is_false_header(line, prev_line, next_line):
    """
    Determine if a line is a false header based on specific criteria.

    Parameters:
        line (str): The line to check.
        prev_line (str): The previous line.
        next_line (str): The next line.

    Returns:
        bool: True if the line is a false header, False otherwise.
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


def split_on_carriage_returns(text):
    """
    Split text into lines separated by double carriage returns.

    Parameters:
        text (str): The input text.

    Returns:
        list: A list of lines.
    """
    lines = re.split(r'\n\n+', re.sub(r':\n', ':\n\n', text))
    lines = [line.strip() for line in lines if line.strip()]
    return lines


def split_lines_on_semicolon(lines):
    """
    Splits lines on semicolons not within braces.

    Parameters:
        lines (list): A list of lines.

    Returns:
        list: A list of split lines.
    """
    split_lines = []
    for line in lines:
        line = replace_parentheses_with_braces(line)
        parts = []
        temp = ""
        inside_braces = False
        for char in line:
            if char == '{':
                inside_braces = True
            elif char == '}':
                inside_braces = False
            elif char == ';' and not inside_braces:
                parts.append(temp.strip())
                temp = ""
                continue
            temp += char
        parts.append(temp.strip())
        split_lines.extend(parts)
    return split_lines


def split_to_sentences(text, regex_patterns, exception_patterns):
    """
    Split text into sentences based on specific criteria.

    Parameters:
        text (str): The input text.
        regex_patterns (list): A list of regex patterns for splitting.
        exception_patterns (list): A list of regex patterns to ignore during splitting.

    Returns:
        list: A list of sentences.
    """
    lines = split_on_carriage_returns(text)
    lines = split_on_leading_markers(lines)
    lines = split_lines_on_semicolon(lines)
    sentences = []

    for line in lines:
        line = re.sub(r"\n", " ", line)
        line = re.sub(' +', ' ', line)
        split_line = split_by_leading_char_from_regex_patterns(
            line, regex_patterns, exceptions_patterns=exception_patterns
        )
        split_line = [s for s in split_line if len(s.split()) > 1]
        sentences.extend(split_line)
        
    return sentences


def drop_leading_character(sentence, regex_patterns):
    """
    Drop leading characters from a sentence based on regex patterns.

    Parameters:
        sentence (str): The input sentence.
        regex_patterns (list): A list of regex patterns.

    Returns:
        str: The cleaned sentence.
    """
    for pattern in regex_patterns:
        while True:
            match = re.match(pattern, sentence)
            if match:
                sentence = re.sub(pattern, '', sentence, count=1).strip()
            else:
                break
    return sentence.strip()


def extract_criteria_sections_headers(lines):
    """
    Extract criteria sub-section headers from a list of lines.

    Parameters:
        lines (list): A list of sentences.

    Returns:
        dict: A dictionary with headers as keys and line indices as values.
    """
    criteria_sections = {}
    # Define explicit patterns for different writing styles of group-specific criteria headers
    patterns = [
        r"^(?:-?\s*)(?:Inclusion|INCLUSION|Exclusion|EXCLUSION|Eligibility|Selection)\s?(?:Criteria|Requirements?)?\s?(?:for|in)?\s?(?:Patients|Subjects|Population|Cohort|Group|Arm)?\s?(?:with|without|who|where|having)?\s?[\w\d\s-]*[:\-]?",
        r"^(?:Key\s)?(?:Inclusion|INCLUSION|EXCLUSION|Exclusion|Eligibility|Selection)(?:\s(?:Criteria|Requirements))?(?:\s?[-+:]|\sfor)?(?:\s[\w\s+-]+)?(?:\([\w\s]+\))?\s?[-+:]?\s?[\w\s]+$",
        r"^(?:Key\s)?(?:Inclusion|INCLUSION|EXCLUSION|Exclusion|Eligibility|Selection)(?:\s(?:Criteria|Requirements?))(?:\s(?:for|in))?(?:\s(?:Patients|Subjects|Population|Cohort|Group|Arm))?(?:\s(?:with|without|who|where|having))?\s?(?:\([\w\s]+\))?\s?[\w\s+-]*[:\-]?",
        r"^(?:[\w\d\s-]+)\s*-\s*(?:Inclusion|INCLUSION|EXCLUSION|Exclusion|Eligibility|Selection)\s(?:Criteria|Requirements?)?$",
        r"^(?:[\w\s]+?)\s(?:group|patients|population|arm|subjects|cohort)\s(?:inclusion|exclusion|eligibility|selection|criteria)(?:\s?:|-)?",
        r"^\b(?:\w+\s\w+|\w+)?\s(?:Inclusion|INCLUSION|EXCLUSION|Exclusion|Eligibility|Selection)\s(?:Criteria|Requirements)\b",
    ]
    for i, line in enumerate(lines):       
        header_candidate = line.strip()
        if ":" in header_candidate:
            header_candidate = header_candidate.split(":")[0].strip()
        if len(header_candidate.split()) <= 10:
            if any(re.search(pattern, header_candidate, re.IGNORECASE) for pattern in patterns): 
                header = header_candidate.strip()
                if header not in criteria_sections:
                    criteria_sections[header] = [i]
                else:
                    criteria_sections[header].append(i)
    return criteria_sections


#################################################
# NEW FUNCTION TO HANDLE INLINE HEADERS
#################################################

def fix_inline_headers(text):
    """
    Ensure that recognized headers (e.g., Inclusion Criteria, Exclusion Criteria, etc.)
    appear on their own line by inserting a newline right after the header phrase
    if it's directly followed by non-whitespace text.
    """
    # Adjust or add more patterns as needed
    patterns = [
        r"(Inclusion\s*Criteria\s*:)\s*(?=\S)",
        r"(Exclusion\s*Criteria\s*:)\s*(?=\S)",
        # Add more if needed
        # r"(Eligibility\s*Criteria\s*:)\s*(?=\S)",
    ]
    
    fixed_text = text
    for pat in patterns:
        fixed_text = re.sub(pat, r"\1\n", fixed_text, flags=re.IGNORECASE)
    return fixed_text


def extract_separate_inclusion_exclusion(text, regex_patterns, exception_patterns):
    """
    Extract preprocessed inclusion and exclusion criteria from eligibility criteria text.

    Parameters:
        text (str): The preprocessed eligibility criteria text.
        regex_patterns (list): A list of regex patterns for splitting.
        exception_patterns (list): A list of regex patterns to ignore during splitting.

    Returns:
        dict: A dictionary containing Inclusion Criteria, Exclusion Criteria, and Original Eligibility Criteria.
    """
    # First, fix the scenario where "Inclusion Criteria:" or "Exclusion Criteria:" 
    # is immediately followed by text on the same line
    text = fix_inline_headers(text)

    criteria = {
        "Inclusion Criteria": {},
        "Exclusion Criteria": {},
        "Original Eligibility Criteria": text
    }
    
    lines = split_to_sentences(text, regex_patterns, exception_patterns)
    subsection_indices = extract_criteria_sections_headers(lines)
    
    # Fallback: If no sections are identified, treat the entire text as one block of inclusion criteria
    if not subsection_indices:
        criteria["Inclusion Criteria"]["General"] = lines
        return criteria
    
    inclusion_pattern = r"(?<!\S)(?:inclusion|INCLUSION|eligibility|selection|included|are eligible)(?!\S|$)"
    exclusion_pattern = r"(?<!\S)(?:exclusion|EXCLUSION|non-inclusion|excluded|not eligible|non-selection)(?!\S|$)"
    
    inclusion_indices = sorted(itertools.chain(*[value for key, value in subsection_indices.items() if re.search(inclusion_pattern, key, re.IGNORECASE)]))
    exclusion_indices = sorted(itertools.chain(*[value for key, value in subsection_indices.items() if re.search(exclusion_pattern, key, re.IGNORECASE)]))
    
    all_indices = sorted([(idx, "Inclusion") for idx in inclusion_indices] + [(idx, "Exclusion") for idx in exclusion_indices])
    
    for i, (start_index, section_type) in enumerate(all_indices):
        end_index = all_indices[i + 1][0] if i + 1 < len(all_indices) else len(lines)
        section_text = lines[start_index + 1:end_index]
        section_header = lines[start_index].strip()
        section_text = [line for line in section_text if line.strip()]  # Remove any empty lines
        
        if section_type == "Inclusion":
            if section_header not in criteria["Inclusion Criteria"]:
                criteria["Inclusion Criteria"][section_header] = []
            criteria["Inclusion Criteria"][section_header].extend(section_text)
        elif section_type == "Exclusion":
            if section_header not in criteria["Exclusion Criteria"]:
                criteria["Exclusion Criteria"][section_header] = []
            criteria["Exclusion Criteria"][section_header].extend(section_text)
    
    return criteria


def split_on_full_stops(text):
    """
    Split text into sentences on actual full stops, avoiding splitting on decimal points and abbreviations.

    Parameters:
        text (str): The input text.

    Returns:
        list: A list of sentences.
    """
    # Pattern to match sentence-ending periods
    pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s'

    # Split the text
    sentences = re.split(pattern, text)

    # Clean up sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def split_large_sentences(df):
    """
    Split any remaining large sentences in the DataFrame on actual full stops.

    Parameters:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        pandas.DataFrame: The DataFrame with large sentences split.
    """
    new_rows = []
    for _, row in df.iterrows():
        sentence = row['sentence']
        if len(sentence) > 200:  # Adjust the threshold as needed
            sentences = split_on_full_stops(sentence)
            for s in sentences:
                if s:  # Ensure the sentence is not empty
                    new_row = row.copy()
                    new_row['sentence'] = s
                    new_rows.append(new_row)
        else:
            new_rows.append(row)
    return pd.DataFrame(new_rows)


def eic_text_preprocessing(_ids, regex_path="../../data/regex/regex_patterns.json", 
                           exceptions_path="../../data/regex/exception_regex_patterns.json", 
                           output_path="../../data/preprocessed_data/clintra/"):
    """
    Main preprocessing function for eligibility criteria text from a list of clinical trial IDs.

    Parameters:
        _ids (list): A list of clinical trial IDs.
        regex_path (str): Path to the regex patterns JSON file.
        exceptions_path (str): Path to the exception regex patterns JSON file.
        output_path (str): Directory path to save the preprocessed CSV file.

    Returns:
        pandas.DataFrame or None: The preprocessed DataFrame or None if no data is processed.
    """
    regex_patterns = list(load_regex_patterns(regex_path).values())
    exception_patterns = list(load_regex_patterns(exceptions_path).values())
    texts = []
    trial_ids = []

    for nid in _ids:
        print(f"Processing Trial ID: {nid}")
        eic_text = extract_eligibility_criteria(nid)
        if eic_text:
            preprocessed_text = extract_separate_inclusion_exclusion(eic_text, regex_patterns, exception_patterns)
            texts.append(preprocessed_text)
            trial_ids.append(nid)
        else:
            continue

    to_concat = []
    for index, item in enumerate(texts):
        _id = trial_ids[index]
        for criteria_key in ["Inclusion Criteria", "Exclusion Criteria"]:
            criteria_dict = item.get(criteria_key, {})
            for sub_key, sub_value in criteria_dict.items():
                df = pd.DataFrame(sub_value, columns=["sentence"])
                df["criteria"] = criteria_key
                df["sub_criteria"] = sub_key
                df["id"] = _id
                to_concat.append(df)

    if to_concat:
        final_df = pd.concat(to_concat, ignore_index=True)
        final_df['sentence'] = final_df['sentence'].apply(drop_leading_character, regex_patterns=regex_patterns)
        final_df['sentence'] = final_df['sentence'].apply(replace_braces_with_parentheses)
        final_df = split_large_sentences(final_df)
        final_df.to_csv(os.path.join(output_path, f"{_ids[0]}_preprocessed.tsv"), index=False, sep='\t')
        return final_df
    else:
        return None
