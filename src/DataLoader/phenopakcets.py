import json
from datetime import datetime

def load_phenopacket(file_path):
    """Load a Phenopacket JSON file."""
    with open(file_path, 'r') as file:
        phenopacket = json.load(file)
    return phenopacket

def convert_iso8601_duration_to_years(iso8601duration):
    """Convert ISO 8601 duration to years (e.g., 'P53Y' to '53 years')."""
    if iso8601duration.startswith('P') and 'Y' in iso8601duration:
        years = iso8601duration.split('Y')[0][1:]  # Extract years
        return f"{years} years"
    return iso8601duration

def extract_age(phenopacket):
    """Extract and format the patient's age."""
    age_info = phenopacket.get("subject", {}).get("age_at_event", {}).get("age", {}).get("iso8601duration")
    if age_info:
        return f"Patient is {convert_iso8601_duration_to_years(age_info)} old"
    return None

def extract_pregnancy_info(phenopacket):
    """Extract pregnancy information if available."""
    gestation = phenopacket.get("subject", {}).get("gestational_age", {})
    weeks = gestation.get("weeks")
    days = gestation.get("days")
    if weeks is not None and days is not None:
        return f"Patient is pregnant. Pregnancy information: {weeks} weeks, {days} days gestation"
    return None

def extract_phenotypic_features(phenopacket):
    """Extract phenotypic features and format them as query strings."""
    return [
        f"{feature['type']['label']}; {feature.get('description', '')}"
        for feature in phenopacket.get("phenotypic_features", [])
    ]

def extract_diseases(phenopacket):
    """Extract disease information and format them as query strings."""
    diseases = []
    for disease in phenopacket.get("diseases", []):
        disease_str = f"{disease['term']['label']}"
        stages = [stage['label'] for stage in disease.get("disease_stage", [])]
        if stages:
            disease_str += f" {', '.join(stages)}"
        if "tumor_progression" in disease:
            disease_str += f", Tumor progression: {disease['tumor_progression']['label']}"
        if "primary_site" in disease:
            disease_str += f", Primary location: {disease['primary_site']['label']}"
        metastatic_sites = [site['label'] for site in disease.get("metastatic_site", [])]
        if metastatic_sites:
            disease_str += f", Metastatic locations: {', '.join(metastatic_sites)}"
        diseases.append(disease_str)
    return diseases

def extract_genomic_information(phenopacket):
    """Extract genomic information and format them as query strings."""
    return [
        f"Gene: {interpretation['gene_descriptor']['symbol']}, "
        f"Variant: {interpretation['variation_descriptor']['hgvs']}, "
        f"Mutation type: {interpretation['variation_descriptor'].get('molecular_consequence', {}).get('label', 'Unknown mutation type')}"
        for interpretation in phenopacket.get("interpretation", {}).get("genomic_interpretations", [])
    ]

def format_duration(start, end):
    """Format the duration between two dates in a readable format."""
    start_date = datetime.fromisoformat(start.rstrip('Z'))
    end_date = datetime.fromisoformat(end.rstrip('Z'))
    duration_days = (end_date - start_date).days
    return f"for the duration of {duration_days} days"

def extract_medical_actions(phenopacket):
    """Extract medical actions and format them as query strings."""
    medical_actions = []
    for action in phenopacket.get("medical_actions", []):
        if "procedure" in action:
            procedure_str = f"Procedure: {action['procedure']['code']['label']}"
            medical_actions.append(procedure_str)

        if "treatment" in action:
            treatment_str = f"Treatment with {action['treatment']['agent']['label']}"
            for dose in action["treatment"].get("dose_intervals", []):
                dosage = dose["quantity"]["value"]
                unit = dose["quantity"]["unit"]["label"]
                frequency = dose["schedule_frequency"]["label"]
                start = dose["interval"]["start"]
                end = dose["interval"]["end"]
                duration = format_duration(start, end)
                treatment_str += f"; dosage: {dosage} {unit}; frequency: {frequency} {duration}"
            medical_actions.append(treatment_str)

        if "radiation_therapy" in action:
            modality = action["radiation_therapy"]["modality"]["label"]
            body_site = action["radiation_therapy"]["body_site"]["label"]
            dose = action["radiation_therapy"]["total_dose"]["value"]
            unit = action["radiation_therapy"]["total_dose"]["unit"]["label"]
            fractions = action["radiation_therapy"]["fractions"]
            radiation_str = (
                f"Radiation therapy using {modality} on {body_site}; total dose: {dose} {unit}, "
                f"delivered in {fractions} fractions."
            )
            medical_actions.append(radiation_str)

        if "therapeutic_regimen" in action:
            regimen_str = f"Therapeutic regimen: {action['therapeutic_regimen']['name']['label']}"
            components = [component["label"] for component in action["therapeutic_regimen"].get("components", [])]
            if components:
                regimen_str += f" including components: {', '.join(components)}"
            medical_actions.append(regimen_str)

        if "outcome" in action:
            action_str = f"Outcome: {action['outcome']}"
            medical_actions.append(action_str)
        elif "summary" in action:
            action_str = f"Summary: {action['summary']}"
            medical_actions.append(action_str)
        elif "description" in action:
            action_str = f"{action['description']}"
            medical_actions.append(action_str)

    return medical_actions

def extract_biosamples(phenopacket):
    """Extract biosample information and format it into sentence-like descriptions."""
    return [
        f"{biosample.get('description', 'A biosample')} was taken from {biosample.get('sampled_tissue', {}).get('label', 'an unspecified tissue')}, "
        f"showing a histological diagnosis of {biosample.get('histological_diagnosis', {}).get('label', 'unknown histological diagnosis')}. "
        f"The tumor was graded as {biosample.get('tumor_grade', {}).get('label', 'unspecified grade')}, "
        f"with progression to {biosample.get('tumor_progression', {}).get('label', 'unknown progression status')} "
        f"and a pathological stage of {biosample.get('pathological_stage', {}).get('label', 'unknown pathological stage')}."
        for biosample in phenopacket.get("biosamples", [])
    ]

def transform_phenopacket_to_queries(phenopacket):
    """Transform a phenopacket into a list of query strings."""
    queries = []
    
    # Extract patient age
    age_query = extract_age(phenopacket)
    if age_query:
        queries.append(age_query)
    
    # Extract pregnancy information if available
    pregnancy_query = extract_pregnancy_info(phenopacket)
    if pregnancy_query:
        queries.append(pregnancy_query)
    
    # Extract other relevant information
    queries.extend(extract_phenotypic_features(phenopacket))
    queries.extend(extract_diseases(phenopacket))
    queries.extend(extract_genomic_information(phenopacket))
    queries.extend(extract_medical_actions(phenopacket))
    queries.extend(extract_biosamples(phenopacket))
    
    return queries

def main():
    # Example usage
    phenopacket_file = '../../data/synthetic_patients/phenopacket13.json'
    phenopacket = load_phenopacket(phenopacket_file)
    queries = transform_phenopacket_to_queries(phenopacket)

    for query in queries:
        print(query)

if __name__ == "__main__":
    main()
