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
    data['condition'] = root.findtext('condition')
    data['intervention'] = { 'intervention_type': root.findtext('intervention/intervention_type'), 'intervention_name': root.findtext('intervention/intervention_name') }
    data['gender'] = root.findtext('eligibility/gender')
    data['minimum_age'] = root.findtext('eligibility/minimum_age')
    data['maximum_age'] = root.findtext('eligibility/maximum_age')
    city = root.findtext('location/facility/address/city')
    state = root.findtext('location/facility/address/state')
    country = root.findtext('location/facility/address/country')

    data['location'] = {
        'location_name': root.findtext('location/facility/name'),
        'location_address': ', '.join(filter(None, [city, state, country]))
    }
    data['reference'] = [{'citation': ref.findtext('citation'), 'PMID': ref.findtext('PMID')} for ref in root.findall('reference')]
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
    input_dir = '../data/trials_xmls/'
    output_dir = '../data/trials_jsons/'
    process_files(input_dir, output_dir)