

import sys
import nltk
import re
import os
import pandas as pd
import utils

INPUT_FILEPATH = "/mnt/cbib/EOSC4Cancer/synthetic_data"
PREPROCESSED_OUTPUT_FILEPATH = "../data/preprocessed_data/patient_notes/"
# remove characters encodings if present
chars = {
    '\xc2\x82' : ',',        # High code comma
    '\xc2\x84' : ',,',       # High code double comma
    '\xc2\x85' : '...',      # Tripple dot
    '\xc2\x88' : '^',        # High carat
    '\xc2\x91' : '\x27',     # Forward single quote
    '\xc2\x92' : '\x27',     # Reverse single quote
    '\xc2\x93' : '\x22',     # Forward double quote
    '\xc2\x94' : '\x22',     # Reverse double quote
    '\xc2\x95' : ' ',
    '\xc2\x96' : '-',        # High hyphen
    '\xc2\x97' : '--',       # Double hyphen
    '\xc2\x99' : ' ',
    '\xc2\xa0' : ' ',
    '\xc2\xa6' : '|',        # Split vertical bar
    '\xc2\xab' : '<<',       # Double less than
    '\xc2\xbb' : '>>',       # Double greater than
    '\xc2\xbc' : '1/4',      # one quarter
    '\xc2\xbd' : '1/2',      # one half
    '\xc2\xbe' : '3/4',      # three quarters
    '\xca\xbf' : '\x27',     # c-single quote
    '\xcc\xa8' : '',         # modifier - under curve
    '\xcc\xb1' : '',          # modifier - under line
    '\x95' : ' ',
}
def replace_chars(match):
    char = match.group(0)
    return chars[char]

def remove_special_chars_encoding(text):
    pattern = '(' + '|'.join(chars.keys()) + ')'
    return re.sub(pattern, replace_chars, text)

def add_space_after_parenthesis(text):
    # Use regular expression to find ")" followed by a letter and insert a space
    result = re.sub(r'\)([a-zA-Z])', r') \1', text)
    return result

def tokenize_clinical_note(id_list, source_directory=INPUT_FILEPATH):
    data = []
    for _id in id_list:
        with open(source_directory + "/" + "%s"%_id + "/" + "clinical_note.txt") as f:
            text = f.read()
            
        phi_tags = re.findall('(\[\*\*.*?\*\*\])', text)
        for i,tag in enumerate(phi_tags):
            text = text.replace(tag, ' __PHI_%d__ ' % i)
            text = text.replace(tag, '__PHI__')

        '''
        Thoughts & Strategies
            - If a newline happens in between matching parens, then ignore newline
            - Identify prose v nonprose. Use nltk.sent_tokenize on prose
                - not perfect, but it is pretty good with ignoring mid-sentence newlines

            - section header: "\n------ Protected Section ------\n"

            - If you can detect a bulleted list, then those are sentences
                - sometimes the bullets are hyphens "-", sometimes numbers "1."

            - ecg reports are very short & seem to be exclusively prose
            - echo reports are VERY structured. definitely useful to do a echo-specific one
            - nursing notes have a linear structure. sections are easily identifiable
            - nursing_other notes are typically very short (and parse-able)
            - radiology reports have lots of "________________________" sections
                - seems like they always have "MEDICAL CONDITION" and "FINAL REPORT" sections
            - found list example where "1." is its own line and its info is own next line :/
                - could be rehab-specific for notes
            - strong consistency format in social_work
        '''

        # break into many segments
        segments = mimic_sent_tokenize_rules(text)

        # run nltk tokenizer on these segments to split prose
        sents = []
        for segment in segments:
            s = nltk.sent_tokenize(segment)
            sents += s


        # put the PHI back
        for i in range(len(sents)):
            tags = re.findall('(__PHI_(\d+)__)', sents[i])
            for tag,ind in tags:
                sents[i] = sents[i].replace(tag, phi_tags[int(ind)])
                
    most_recent_title = None
    most_recent_inline_title = None

    for sentence in sents:
        sentence = remove_full_stops_and_commas(sentence)
        
        if len(sentence) > 1:
            is_sentence_title = is_title(sentence)
            is_sentence_inline_title = is_inline_title(sentence)
            
            field = None
            
            if is_sentence_title:
                most_recent_title = sentence
                field = sentence[:-1]
                sentence = "___"
            elif is_sentence_inline_title:
                most_recent_inline_title = sentence.split(":", 1)[0]
                field = most_recent_inline_title
                sentence = sentence.split(":", 1)[1]
            else:
                # Check for the most recent title or inline title encountered
                if most_recent_title:
                    field = most_recent_title[:-1]
                elif most_recent_inline_title:
                    field = most_recent_inline_title
            
            data.append((_id, field, sentence.replace("___", ""), is_sentence_title, is_sentence_inline_title))
        else:
            continue

        # Create a DataFrame with 'Sentence' and 'IsTitle' columns
    df = pd.DataFrame(data, columns=['id', 'field', 'sentence', 'IsTitle', 'IsInlineTitle'])
    df.to_csv(PREPROCESSED_OUTPUT_FILEPATH + "%s_preprocessed.csv"%id_list[0])
    return df


def mimic_sent_tokenize_rules(text):

    # long sections are OBVIOUSLY different sentences
    text = re.sub('---+', '\n\n-----\n\n', text)
    text = re.sub('___+', '\n\n_____\n\n', text)
    text = re.sub('\n\n+', '\n\n', text)
    text = text.replace("___", "")
    text = re.sub(r'(?<!\n)\n(?!\n)(?=[A-Z])', '\n\n', text)
    text = re.sub(r'  ', ' ', text)
    text = add_space_after_parenthesis(text)
    text = remove_special_chars_encoding(text)
    
    segments = re.split(r'\n\n+', re.sub(r':\n', ':\n\n', text))

    # strategy: break down segments and chip away structure until just prose.
    #           once you have prose, use nltk.sent_tokenize()

    ### Separate section headers ###
    new_segments = []

    # deal with this one edge case (multiple headers per line) up front
    m1 = re.match('(Admission Date:) (.*) (Discharge Date:) (.*)', segments[0])
    if m1:
        new_segments += list(map(strip,m1.groups()))
        segments = segments[1:]

    m2 = re.match('(Date of Birth:) (.*) (Sex:) (.*)'            , segments[0])
    if m2:
        new_segments += list(map(strip,m2.groups()))
        segments = segments[1:]

    for segment in segments:
        # find all section headers
        possible_headers  = re.findall('\n([A-Z][^\n:]+:)', '\n'+segment)
        #assert len(possible_headers) < 2, str(possible_headers)
        headers = []
        for h in possible_headers:
            #print('cand=[%s]' % h
            if is_title(h.strip()):
                #print('\tYES=[%s]' % h
                headers.append(h.strip())

        # split text into new segments, delimiting on these headers
        for h in headers:
            h = h.strip()

            # split this segment into 3 smaller segments
            ind = segment.index(h)
            prefix = segment[:ind].strip()
            rest   = segment[ ind+len(h):].strip()

            # add the prefix (potentially empty)
            if len(prefix) > 0:
                new_segments.append(prefix.strip())

            # add the header
            new_segments.append(h)

            # remove the prefix from processing (very unlikely to be empty)
            segment = rest.strip()

        # add the final piece (aka what comes after all headers are processed)
        if len(segment) > 0:
            new_segments.append(segment.strip())

    # copy over the new list of segments (further segmented than original segments)
    segments = list(new_segments)
    new_segments = []


    ### Low-hanging fruit: "_____" is a delimiter
    for segment in segments:
        subsections = segment.split('\n_____\n')
        new_segments.append(subsections[0])
        for ss in subsections[1:]:
            new_segments.append('_____')
            new_segments.append(ss)

    segments = list(new_segments)
    new_segments = []


    ### Low-hanging fruit: "-----" is a delimiter
    for segment in segments:
        subsections = segment.split('\n-----\n')
        new_segments.append(subsections[0])
        for ss in subsections[1:]:
            new_segments.append('-----')
            new_segments.append(ss)

    segments = list(new_segments)
    new_segments = []

    '''
    for segment in segments:
        print('------------START------------')
        print(segment)
        print('-------------END-------------')
        print
    exit()
    '''

    ### Separate enumerated lists ###
    for segment in segments:
        if not re.search('\n\s*\d+\.', '\n'+segment): 
            new_segments.append(segment)
            continue

        '''
        print('------------START------------')
        print(segment)
        print('-------------END-------------')
        print
        '''

        # generalizes in case the list STARTS this section
        segment = '\n'+segment

        # determine whether this segment contains a bulleted list (assumes i,i+1,...,n)
        start = int(re.search('\n\s*(\d+)\.', '\n'+segment).groups()[0])
        n = start
        while re.search('\n\s*%d.'%n,segment):
            n += 1
        n -= 1

        # no bulleted list
        if n < 1:
            new_segments.append(segment)
            continue

        '''
        print('------------START------------')
        print(segment)
        print('-------------END-------------')

        print(start,n)
        print
        '''

        # break each list into its own line
        # challenge: not clear how to tell when the list ends if more text happens next
        for i in range(start,n+1):
            matching_text = re.search('(\n\s*\d+\.)',segment)
            if matching_text:
                matching_text = matching_text.groups()[0]
                prefix  = segment[:segment.index(matching_text) ].strip()
                segment = segment[ segment.index(matching_text):].strip()
            else:
                continue
            
            if len(prefix)>0:
                new_segments.append(prefix)

        if len(segment)>0:
            new_segments.append(segment)

    segments = list(new_segments)
    new_segments = []

    '''
        TODO: Big Challenge

        There is so much variation in what makes a list. Intuitively, I can tell it's a
        list because it shows repeated structure (often following a header)

        Examples of some lists (with numbers & symptoms changed around to noise)

            Past Medical History:
            -- Hyperlipidemia
            -- lactose intolerance
            -- Hypertension


            Physical Exam:
            Vitals - T 82.2 BP 123/23 HR 73 R 21 75% on 2L NC
            General - well appearing male, sitting up in chair in NAD
            Neck - supple, JVP elevated to angle of jaw 
            CV - distant heart sounds, RRR, faint __PHI_43__ murmur at


            Labs:
            __PHI_10__ 12:00PM BLOOD WBC-8.8 RBC-8.88* Hgb-88.8* Hct-88.8*
            MCV-88 MCH-88.8 MCHC-88.8 RDW-88.8* Plt Ct-888
            __PHI_14__ 04:54AM BLOOD WBC-8.8 RBC-8.88* Hgb-88.8* Hct-88.8*
            MCV-88 MCH-88.8 MCHC-88.8 RDW-88.8* Plt Ct-888
            __PHI_23__ 03:33AM BLOOD WBC-8.8 RBC-8.88* Hgb-88.8* Hct-88.8*
            MCV-88 MCH-88.8 MCHC-88.8 RDW-88.8* Plt Ct-888
            __PHI_109__ 03:06AM BLOOD WBC-8.8 RBC-8.88* Hgb-88.8* Hct-88.8*
            MCV-88 MCH-88.8 MCHC-88.8 RDW-88.8* Plt Ct-888
            __PHI_1__ 05:09AM BLOOD WBC-8.8 RBC-8.88* Hgb-88.8* Hct-88.8*
            MCV-88 MCH-88.8 MCHC-88.8 RDW-88.8* Plt Ct-888
            __PHI_26__ 04:53AM BLOOD WBC-8.8 RBC-8.88* Hgb-88.8* Hct-88.8*
            MCV-88 MCH-88.8 MCHC-88.8 RDW-88.8* Plt Ct-888
            __PHI_301__ 05:30AM BLOOD WBC-8.8 RBC-8.88* Hgb-88.8* Hct-88.8*
            MCV-88 MCH-88.8 MCHC-88.8 RDW-88.8* Plt Ct-888


            Medications on Admission:
            Allopurinol 100 mg DAILY
            Aspirin 250 mg DAILY
            Atorvastatin 10 mg DAILY
            Glimepiride 1 mg once a week.
            Hexavitamin DAILY
            Lasix 50mg M-W-F; 60mg T-Th-Sat-Sun
            Metoprolol 12.5mg TID
            Prilosec OTC 20 mg once a day
            Verapamil 120 mg SR DAILY
    '''

    ### Remove lines with inline titles from larger segments (clearly nonprose)
    for segment in segments:
        '''
        With: __PHI_6__, MD __PHI_5__
        Building: De __PHI_45__ Building (__PHI_32__ Complex) __PHI_87__
        Campus: WEST
        '''

        lines = segment.split('\n')

        buf = []
        for i in range(len(lines)):
            if is_inline_title(lines[i]):
                if len(buf) > 0:
                    new_segments.append('\n'.join(buf))
                buf = []
            buf.append(lines[i])
        if len(buf) > 0:
            new_segments.append('\n'.join(buf))

    segments = list(new_segments)
    new_segments = []
    # Going to put one-liner answers with their sections 
    # (aka A A' B B' C D D' -->  AA' BB' C DD' )
    N = len(segments)
    for i in range(len(segments)):
        # avoid segfaults
        if i==0: 
            new_segments.append(segments[i])
            continue

        if     segments[i].count('\n') == 0       and \
               is_title(segments[i-1]) and \
           not is_title(segments[i  ]):
            if (i == N-1) or is_title(segments[i+1]):
                new_segments = new_segments[:-1]
                new_segments.append(segments[i-1] + ' ' + segments[i])
        else:
            new_segments.append(segments[i].replace("\n", " "))

    segments = list(new_segments)
    new_segments = []

    '''
        Should do some kind of regex to find "TEST: value" in segments?

            Indication: Source of embolism.
            BP (mm Hg): 145/89
            HR (bpm): 80

        Note: I made a temporary hack that fixes this particular problem. 
              We'll see how it shakes out
    '''
    
    '''
        Separate ALL CAPS lines (Warning... is there ever prose that can be all caps?)
    '''

    '''
    for segment in segments:
        print('------------START------------')
        print(segment)
        print('-------------END-------------')
        print
    exit()
    '''

    return segments



def strip(s):
    return s.strip()



def is_inline_title(text):
    m = re.search('^([a-zA-Z ]+:) ', text)
    if not m:
        return False

    return is_title(m.groups()[0])

def remove_full_stops_and_commas(text):
    text = text.replace(',', '')
    if text.endswith('.'):
        return text[:-1]  # Return text without the last character (full stop)
    return text 

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


stopwords = set(['of', 'on', 'or'])
def is_title(text):
    if not text.endswith(':'):
        return False
    text = text[:-1]

    # be a little loose here... can tighten if it causes errors
    text = re.sub('(\([^\)]*?\))', '', text)

    # Are all non-stopwords capitalized?
    for word in text.split():
        if word in stopwords: continue
        if not word[0].isupper():
            return False

    # I noticed this is a common issue (non-title aapears at beginning of line)
    if text == 'Disp':
        return False

    # optionally: could assert that it is less than 6 tokens

    return True

def main():

    '''
    # read text file from command line
    if len(sys.argv) != 2:
        print(>>sys.stderr, '\n\tusage: python %s <txt-file>\n' % sys.argv[0]
        exit(1)
    mimic_note_file = sys.argv[1]

    with open(mimic_note_file, 'r') as f:
        text = f.read()

    # tokenize
    sents = tokenize_clinical_note(text)
    '''

    for mimic_note_file in sys.argv[1:]:
        print(mimic_note_file)

        with open(mimic_note_file, 'r') as f:
            text = f.read()

        # tokenize
        sents = tokenize_clinical_note(text)

        for sent in sents:
            print('-'*40)
            print(sent)
            print('='*40)
            print('\n\n')


if __name__ == '__main__':
    main()


