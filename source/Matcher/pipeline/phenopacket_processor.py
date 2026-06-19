import json
from typing import Dict, List, Optional

import torch
from Matcher.schemas.phenopacket import Phenopacket
from Matcher.utils.file_utils import read_json_file, write_json_file
from Matcher.utils.json_utils import extract_json_object
from Matcher.utils.logging_config import setup_logging
from Matcher.utils.temporal_utils import parse_iso_duration, parse_temporal
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = setup_logging(__name__)


class PhenopacketProcessor:
    def __init__(self, file_path: str):
        self.phenopacket = self._load_and_validate(file_path)
        self.medical_sentences: List[str] = []
        self.ontology_cache: Dict[str, str] = {}

    def _load_and_validate(self, file_path: str) -> Dict:
        data = read_json_file(file_path)
        try:
            Phenopacket.model_validate(data)
            logger.info("Phenopacket loaded and validated successfully.")
        except Exception as exc:
            raise ValueError(f"Invalid Phenopacket: {exc}") from exc
        return data

    def _add_medical_sentence(self, category: str, content: str):
        self.medical_sentences.append(f"{category.upper()}: {content}")

    def _get_ontology_label(self, term_id: str) -> str:
        if term_id in self.ontology_cache:
            return self.ontology_cache[term_id]
        for resource in self.phenopacket.get("metaData", {}).get("resources", []):
            if term_id.startswith(resource.get("namespacePrefix", "")):
                label = term_id.split("/")[-1].replace("_", " ")
                self.ontology_cache[term_id] = label
                return label
        return term_id

    def generate_medical_narrative(self) -> List[str]:
        try:
            self._extract_subject()
            self._extract_phenotypic_features()
            self._extract_diseases()
            self._extract_biosamples()
            self._extract_interpretations()
            self._extract_medical_actions()
            self._extract_measurements()
            self._extract_family()
            logger.info("Medical narrative generation completed successfully.")
            return self.medical_sentences
        except Exception as e:
            logger.error(f"Narrative generation failed: {e}")
            raise

    def _extract_subject(self):
        subject = self.phenopacket.get("subject", {})
        demographics = [
            f"Sex: {subject.get('sex', 'Unknown')}",
            f"DOB: {subject.get('dateOfBirth', 'Unknown')}",
        ]
        if "timeAtLastEncounter" in subject:
            encounter_time = parse_temporal(subject.get("timeAtLastEncounter"))
            demographics.append(f"Last Encounter: {encounter_time}")
        if "taxonomy" in subject:
            demographics.append(
                f"Species: {subject['taxonomy'].get('label', 'Unknown')}"
            )
        if subject.get("description"):
            demographics.append(f"Description: {subject.get('description')}")
        self._add_medical_sentence("DEMOGRAPHICS", "; ".join(demographics))

    def _extract_phenotypic_features(self):
        for pf in self.phenopacket.get("phenotypicFeatures", []):
            feature = pf.get("type", {})
            label = self._get_ontology_label(feature.get("label", "Unknown"))
            details = ["Absent" if pf.get("excluded", False) else "Present"]
            if "severity" in pf:
                severity = self._get_ontology_label(
                    pf["severity"].get("label", "Unknown")
                )
                details.append(f"Severity: {severity}")
            temporal = parse_temporal(pf.get("onset"))
            if temporal:
                details.append(f"Onset: {temporal}")
            if "modifiers" in pf:
                mods = [
                    self._get_ontology_label(m.get("label", "Unknown"))
                    for m in pf["modifiers"]
                ]
                details.append(f"Modifiers: {', '.join(mods)}")
            descs = []
            if pf.get("description"):
                descs.append(pf.get("description"))
            if feature.get("description"):
                descs.append(feature.get("description"))
            if descs:
                details.append("Description: " + " ".join(descs))
            self._add_medical_sentence("PHENOTYPE", f"{label}: {'; '.join(details)}")

    def _extract_diseases(self):
        for disease in self.phenopacket.get("diseases", []):
            term = disease.get("term", {})
            label = self._get_ontology_label(term.get("label", "Unknown"))
            details = []
            if "diseaseStage" in disease:
                stages = [
                    self._get_ontology_label(s.get("label", "Unknown"))
                    for s in disease["diseaseStage"]
                ]
                details.append(f"Stage: {', '.join(stages)}")
            if "tnmFinding" in disease:
                tnm = [
                    self._get_ontology_label(t.get("label", "Unknown"))
                    for t in disease["tnmFinding"]
                ]
                details.append(f"TNM: {', '.join(tnm)}")
            onset = parse_temporal(disease.get("onset"))
            if onset:
                details.append(f"Onset: {onset}")
            if disease.get("description"):
                details.append(f"Description: {disease.get('description')}")
            self._add_medical_sentence("DIAGNOSIS", f"{label}: {'; '.join(details)}")

    def _extract_biosamples(self):
        for sample in self.phenopacket.get("biosamples", []):
            details = [
                f"Type: {self._get_ontology_label(sample.get('sampleType', {}).get('label', 'Unknown'))}",
                f"Tissue: {self._get_ontology_label(sample.get('sampledTissue', {}).get('label', 'Unknown'))}",
                f"Collection: {parse_temporal(sample.get('timeOfCollection'))}",
            ]
            if "histologicalDiagnosis" in sample:
                hd = sample["histologicalDiagnosis"]
                details.append(
                    f"Histology: {self._get_ontology_label(hd.get('label', 'Unknown'))}"
                )
            if sample.get("description"):
                details.append(f"Description: {sample.get('description')}")
            self._add_medical_sentence("BIOSAMPLE", "; ".join(details))

    def _extract_measurements(self):
        for meas in self.phenopacket.get("measurements", []):
            assay = self._get_ontology_label(meas.get("assay", {}).get("id", "Unknown"))
            value_info = meas.get("value", {})
            unit_label = value_info.get("unit", {}).get("label", "")
            value = f"{value_info.get('value', 'Unknown')} {unit_label}".strip()
            content = f"{assay}: {value}"
            if meas.get("description"):
                content += f"; Description: {meas.get('description')}"
            self._add_medical_sentence("MEASUREMENT", content)

    def _extract_medical_actions(self):
        for action in self.phenopacket.get("medicalActions", []):
            if "treatment" in action:
                tx = action["treatment"]
                details = [
                    f"Agent: {self._get_ontology_label(tx.get('agent', {}).get('label', 'Unknown'))}",
                    f"Route: {tx.get('routeOfAdministration', {}).get('label', 'Unknown')}",
                ]
                if "doseIntervals" in tx and tx["doseIntervals"]:
                    dose_info = tx["doseIntervals"][0].get("quantity", {})
                    dose_unit = dose_info.get("unit", {}).get("label", "")
                    details.append(
                        f"Dose: {dose_info.get('value', 'Unknown')} {dose_unit}".strip()
                    )
                if tx.get("description"):
                    details.append(f"Description: {tx.get('description')}")
                self._add_medical_sentence("TREATMENT", "; ".join(details))
            if "procedure" in action:
                proc = action["procedure"]
                performed = proc.get("performed", "unknown date")
                proc_details = (
                    f"{proc.get('code', {}).get('label', 'Procedure')} on {performed}"
                )
                if proc.get("description"):
                    proc_details += f"; Description: {proc.get('description')}"
                self._add_medical_sentence("PROCEDURE", proc_details)

    def _extract_interpretations(self):
        for interpret in self.phenopacket.get("interpretations", []):
            if "diagnosis" not in interpret:
                continue
            dx = interpret["diagnosis"]
            details = [
                f"Status: {dx.get('diagnosisStatus', {}).get('label', 'unknown')}"
            ]
            if dx.get("description"):
                details.append(f"Description: {dx.get('description')}")
            for gi in dx.get("genomicInterpretations", []):
                if "variantInterpretation" in gi:
                    var = gi["variantInterpretation"].get("variationDescriptor", {})
                    gene = var.get("geneContext", {}).get("symbol", "Unknown gene")
                    details.append(f"{gene} {var.get('label', 'variant')}")
            if interpret.get("description"):
                details.append(f"Note: {interpret.get('description')}")
            self._add_medical_sentence("INTERPRETATION", "; ".join(details))

    def _extract_family(self):
        family = self.phenopacket.get("family")
        if not family:
            return
        relatives = family.get("relatives", [])
        for relative in relatives:
            rel_id = relative.get("id", "Unknown")
            sex = relative.get("sex", "Unknown")
            vital_status = relative.get("vitalStatus", {}).get("status", "Unknown")
            age_at_death = parse_iso_duration(
                relative.get("vitalStatus", {})
                .get("ageAtDeath", {})
                .get("iso8601duration", "")
            )
            rel_phens = []
            for pf in relative.get("phenotypicFeatures", []):
                label = self._get_ontology_label(
                    pf.get("type", {}).get("label", "Unknown")
                )
                onset = parse_temporal(pf.get("onset"))
                rel_phens.append(f"{label} (onset: {onset})")
            description = relative.get("description", "")
            summary = f"Relative {rel_id} ({sex}) - Vital status: {vital_status}"
            if age_at_death:
                summary += f", Age at death: {age_at_death}"
            if rel_phens:
                summary += f"; Phenotypes: {', '.join(rel_phens)}"
            if description:
                summary += f"; Description: {description}"
            self._add_medical_sentence("FAMILY_HISTORY", summary)
            pedigree = family.get("pedigree", {})
            if pedigree.get("persons"):
                summary = (
                    f"Pedigree defined with {len(pedigree['persons'])} individuals."
                )
                self._add_medical_sentence("FAMILY_PEDIGREE", summary)


class ClinicalSummarizer:
    def __init__(self, model=None, tokenizer=None, model_name: Optional[str] = None):
        if model is not None:
            if tokenizer is None:
                raise ValueError(
                    "A tokenizer must be provided if a model instance is given."
                )
            self.model = model
            self.tokenizer = tokenizer
        elif model_name is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        else:
            raise ValueError(
                "Must provide either a model instance with its tokenizer or a model_name."
            )

        self.model.eval()

    def generate_summary(self, sentences: List[str]) -> Dict:
        SYSTEM_PROMPT = """
        You are a specialized medical assistant designed for precise and accurate clinical trial matching.
        Analyze the patient's medical description carefully and extract clinically relevant information for trial eligibility assessment.

        1. **Primary Condition**:
            - Determine the primary medical conditions based on explicit patient information and overall clinical context.
            - List up to 10 medically recognized synonyms, aliases, or closely related medical terms for the primary conditions.
            - Include the identified primary conditions and their associated synonyms or related terms within the "main_conditions" list.

        2. **Secondary Clinical Factors**:
            - Provide up to 50 clinically significant additional factors, including comorbidities, concurrent medical conditions, molecular or genetic biomarkers, prior therapies, relevant medical history, and clinically notable patient characteristics explicitly mentioned in the patient description.
            - Provide these factors in the "other_conditions" list.

        3. **Expanded Clinical Descriptions**:
            - Based solely on the original patient-provided data, generate semantically accurate and medically sound statements resembling real-life medical notes.
            - **Crucial**: Expanded descriptions must strictly reflect explicit patient-reported information without introducing new or inferred medical details.

        Output:
        Return a JSON object in the exact following structure without any additional commentary:

        {
        "main_conditions": ["PrimaryCondition", "Synonym1", "Synonym2", "..."],
        "other_conditions": ["AdditionalCondition1", "AdditionalCondition2", "..."],
        "expanded_sentences": [
            "Expanded note for sentence 1...",
            "Expanded note for sentence 2...",
            "..."
        ]
        }
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": " ".join(sentences)},
        ]

        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                truncate=False,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    prompt,
                    max_new_tokens=2048,
                    do_sample=False,
                    return_dict_in_generate=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Only decode the newly generated tokens
            generated_text = self.tokenizer.decode(
                output_ids[0][prompt.shape[-1] :], skip_special_tokens=True
            )
            return self._extract_llm_output(generated_text)

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return {
                "main_conditions": [],
                "other_conditions": [],
                "expanded_sentences": [],
                "error": str(e),
            }

    def _extract_llm_output(self, generated_text: str) -> Dict:
        try:
            result = extract_json_object(generated_text)
            expected_keys = [
                "main_conditions",
                "other_conditions",
                "expanded_sentences",
            ]
            if not all(key in result for key in expected_keys):
                raise ValueError("Missing required JSON keys in LLM output")
            return result
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"JSON extraction failed: {e}")
            return {
                "error": f"Invalid JSON format: {str(e)}",
                "main_conditions": [],
                "other_conditions": [],
                "expanded_sentences": [],
            }


def process_phenopacket(
    input_file: str,
    output_file: str,
    model=None,
    tokenizer=None,
    model_name: str = "microsoft/phi-2",
) -> bool:
    try:
        processor = PhenopacketProcessor(input_file)
        narrative = processor.generate_medical_narrative()
        summarizer = (
            ClinicalSummarizer(model=model, tokenizer=tokenizer)
            if model and tokenizer
            else ClinicalSummarizer(model_name=model_name)
        )
        summary = summarizer.generate_summary(narrative)
        write_json_file(summary, output_file)
        logger.info(
            f"Successfully processed {input_file} and saved results to {output_file}"
        )
        return True
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return False
