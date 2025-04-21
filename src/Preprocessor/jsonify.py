import xml.etree.ElementTree as ET
import json
import os

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    data = {}

    # Extracting required fields
    data['nct_id'] = root.findtext('id_info/nct_id')
    data['brief_title'] = root.findtext('brief_title')
    data['official_title'] = root.findtext('official_title')
    data['brief_summary'] = root.findtext('brief_summary/textblock')
    data['detailed_description'] = root.findtext('detailed_description/textblock')
    data['overall_status'] = root.findtext('overall_status')
    data['start_date'] = root.findtext('start_date')
    data['completion_date'] = root.findtext('completion_date')
    data['phase'] = root.findtext('phase')
    data['study_type'] = root.findtext('study_type')

    # Handle multiple conditions
    data['condition'] = [cond.text for cond in root.findall('condition')]

    # Handle multiple interventions
    data['intervention'] = []
    for intervention in root.findall('intervention'):
        data['intervention'].append({
            'intervention_type': intervention.findtext('intervention_type'),
            'intervention_name': intervention.findtext('intervention_name')
        })

    data['gender'] = root.findtext('eligibility/gender')
    data['minimum_age'] = root.findtext('eligibility/minimum_age')
    data['maximum_age'] = root.findtext('eligibility/maximum_age')

    # Extract eligibility criteria as a single block
    data['eligibility_criteria'] = root.findtext('eligibility/criteria/textblock')

    # Handle multiple locations
    data['location'] = []
    for location in root.findall('location'):
        city = location.findtext('facility/address/city')
        state = location.findtext('facility/address/state')
        country = location.findtext('facility/address/country')
        location_name = location.findtext('facility/name')
        location_address = ', '.join(filter(None, [city, state, country]))
        data['location'].append({
            'location_name': location_name,
            'location_address': location_address
        })

    # Handle multiple references
    data['reference'] = []
    for ref in root.findall('reference'):
        data['reference'].append({
            'citation': ref.findtext('citation'),
            'PMID': ref.findtext('PMID')
        })

    return data

def convert_to_json(data):
    json_data = json.dumps(data, indent=4)
    return json_data

def process_files(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith('.xml'):
            xml_file = os.path.join(input_dir, filename)
            data = parse_xml(xml_file)
            json_data = convert_to_json(data)

            # Save JSON to output directory
            json_file = os.path.join(output_dir, filename.replace('.xml', '.json'))
            with open(json_file, 'w') as f:
                f.write(json_data)

if __name__ == "__main__":
    input_dir = '../../data/trials_xmls/'
    output_dir = '../../data/trials_jsons/'
    process_files(input_dir, output_dir)
