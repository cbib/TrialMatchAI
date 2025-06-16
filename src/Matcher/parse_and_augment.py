import json
import logging
import re
import sys
from datetime import datetime
from typing import Dict, List, Optional
from transformers import pipeline, AutoTokenizer
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


class PhenopacketProcessor:
    """Comprehensive processor for Phenopacket v2.x with full schema support."""

    def __init__(self, file_path: str) -> None:
        self.phenopacket: Dict = self._load_and_validate(file_path)
        self.medical_sentences: List[str] = []
        self.ontology_cache: Dict[str, str] = {}

    def _load_and_validate(self, file_path: str) -> Dict:
        """Load and validate the Phenopacket structure."""
        with open(file_path, "r") as f:
            data = json.load(f)

        required_fields = ["id", "metaData", "subject"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Invalid Phenopacket: Missing required field {field}")
        logging.info("Phenopacket loaded and validated successfully.")
        return data

    def _add_medical_sentence(self, category: str, content: str) -> None:
        """Format and store medical information without including medical codes."""
        sentence = f"{category.upper()}: {content}"
        self.medical_sentences.append(sentence)

    def _parse_temporal(self, temporal_obj: Optional[Dict]) -> str:
        """Parse complex temporal elements with error handling."""
        if not temporal_obj:
            return "Timing not specified"

        try:
            if "age" in temporal_obj:
                return self._parse_iso_duration(
                    temporal_obj["age"].get("iso8601duration")
                )
            if "timestamp" in temporal_obj:
                return datetime.fromisoformat(temporal_obj["timestamp"]).strftime(
                    "%Y-%m-%d"
                )
            if "interval" in temporal_obj:
                start = temporal_obj["interval"].get("start", "unknown")
                end = temporal_obj["interval"].get("end", "unknown")
                return f"{start} to {end}"
            return "Timing information available"
        except Exception as e:
            logging.warning(f"Temporal parsing error: {e}")
            return "Timing information unavailable"

    def _parse_iso_duration(self, duration: Optional[str]) -> str:
        """Convert ISO8601 duration to a human-readable format."""
        if not duration:
            return "Age unspecified"

        try:
            match = re.match(r"P(?:(\d+)Y)?(?:(\d+)M)?(?:(\d+)D)?", duration)
            parts = []
            if match:
                if match.group(1):
                    parts.append(f"{match.group(1)} years")
                if match.group(2):
                    parts.append(f"{match.group(2)} months")
                if match.group(3):
                    parts.append(f"{match.group(3)} days")
                return " ".join(parts) if parts else duration
            return duration
        except Exception as e:
            logging.warning(f"Duration parsing failed: {e}")
            return duration

    def _get_ontology_label(self, term_id: str) -> str:
        """Resolve ontology terms using metadata resources."""
        if term_id in self.ontology_cache:
            return self.ontology_cache[term_id]

        for resource in self.phenopacket.get("metaData", {}).get("resources", []):
            if term_id.startswith(resource.get("namespacePrefix", "")):
                # Use only the human-readable label
                label = term_id.split("/")[-1].replace("_", " ")
                self.ontology_cache[term_id] = label
                return label

        return term_id

    def generate_medical_narrative(self) -> List[str]:
        """Generate a complete clinical narrative from the Phenopacket."""
        try:
            self._extract_subject()
            self._extract_phenotypic_features()
            self._extract_diseases()
            self._extract_biosamples()
            self._extract_interpretations()
            self._extract_medical_actions()
            self._extract_measurements()
            self._extract_family()
            logging.info("Medical narrative generation completed successfully.")
            return self.medical_sentences
        except Exception as e:
            logging.error(f"Narrative generation failed: {e}")
            raise

    def _extract_subject(self) -> None:
        """Extract demographic and taxonomic information."""
        subject = self.phenopacket.get("subject", {})

        demographics = [
            f"Sex: {subject.get('sex', 'Unknown')}",
            f"DOB: {subject.get('dateOfBirth', 'Unknown')}",
        ]

        if "timeAtLastEncounter" in subject:
            encounter_time = self._parse_temporal(subject.get("timeAtLastEncounter"))
            demographics.append(f"Last Encounter: {encounter_time}")

        if "taxonomy" in subject:
            tax = subject["taxonomy"]
            demographics.append(f"Species: {tax.get('label', 'Unknown')}")

        # Capture optional description for the subject if it exists.
        if subject.get("description"):
            demographics.append(f"Description: {subject.get('description')}")

        self._add_medical_sentence("DEMOGRAPHICS", "; ".join(demographics))

    def _extract_phenotypic_features(self) -> None:
        """Parse phenotypic observations with any modifiers."""
        for pf in self.phenopacket.get("phenotypicFeatures", []):
            feature = pf.get("type", {})
            label = self._get_ontology_label(feature.get("label", "Unknown"))

            details = []
            if pf.get("excluded", False):
                details.append("Absent")
            else:
                details.append("Present")

            if "severity" in pf:
                severity = self._get_ontology_label(
                    pf["severity"].get("label", "Unknown")
                )
                details.append(f"Severity: {severity}")

            temporal = self._parse_temporal(pf.get("onset"))
            if temporal:
                details.append(f"Onset: {temporal}")

            if "modifiers" in pf:
                mods = [
                    self._get_ontology_label(m.get("label", "Unknown"))
                    for m in pf["modifiers"]
                ]
                details.append(f"Modifiers: {', '.join(mods)}")

            # Capture any description found on the feature level or the phenotypic observation itself.
            descs = []
            if pf.get("description"):
                descs.append(pf.get("description"))
            if feature.get("description"):
                descs.append(feature.get("description"))
            if descs:
                details.append("Description: " + " ".join(descs))

            self._add_medical_sentence("PHENOTYPE", f"{label}: {'; '.join(details)}")

    def _extract_diseases(self) -> None:
        """Extract disease diagnoses along with staging and TNM findings."""
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

            onset = self._parse_temporal(disease.get("onset"))
            if onset:
                details.append(f"Onset: {onset}")

            if disease.get("description"):
                details.append(f"Description: {disease.get('description')}")

            self._add_medical_sentence("DIAGNOSIS", f"{label}: {'; '.join(details)}")

    def _extract_biosamples(self) -> None:
        """Process biosample data, including histopathology details."""
        for sample in self.phenopacket.get("biosamples", []):
            details = [
                f"Type: {self._get_ontology_label(sample.get('sampleType', {}).get('label', 'Unknown'))}",
                f"Tissue: {self._get_ontology_label(sample.get('sampledTissue', {}).get('label', 'Unknown'))}",
                f"Collection: {self._parse_temporal(sample.get('timeOfCollection'))}",
            ]

            if "histologicalDiagnosis" in sample:
                hd = sample["histologicalDiagnosis"]
                details.append(
                    f"Histology: {self._get_ontology_label(hd.get('label', 'Unknown'))}"
                )

            if sample.get("description"):
                details.append(f"Description: {sample.get('description')}")

            self._add_medical_sentence("BIOSAMPLE", "; ".join(details))

    def _extract_measurements(self) -> None:
        """Handle clinical measurements and biomarker data."""
        for meas in self.phenopacket.get("measurements", []):
            assay = self._get_ontology_label(meas.get("assay", {}).get("id", "Unknown"))
            value_info = meas.get("value", {})
            unit_label = value_info.get("unit", {}).get("label", "")
            value = f"{value_info.get('value', 'Unknown')} {unit_label}".strip()
            content = f"{assay}: {value}"
            if meas.get("description"):
                content += f"; Description: {meas.get('description')}"
            self._add_medical_sentence("MEASUREMENT", content)

    def _extract_medical_actions(self) -> None:
        """Process treatments and procedures."""
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

    def _extract_interpretations(self) -> None:
        """Analyze genomic interpretations and related details."""
        for interpret in self.phenopacket.get("interpretations", []):
            if "diagnosis" not in interpret:
                continue

            dx = interpret["diagnosis"]
            details = [
                f"Status: {dx.get('diagnosisStatus', {}).get('label', 'unknown')}"
            ]

            # Capture description at the diagnosis level if available.
            if dx.get("description"):
                details.append(f"Description: {dx.get('description')}")

            for gi in dx.get("genomicInterpretations", []):
                if "variantInterpretation" in gi:
                    var = gi["variantInterpretation"].get("variationDescriptor", {})
                    gene = var.get("geneContext", {}).get("symbol", "Unknown gene")
                    details.append(f"{gene} {var.get('label', 'variant')}")

            # Also capture any overarching interpretation description
            if interpret.get("description"):
                details.append(f"Note: {interpret.get('description')}")

            self._add_medical_sentence("INTERPRETATION", "; ".join(details))

    def _extract_family(self) -> None:
        """Extract family history and pedigree data."""
        family = self.phenopacket.get("family")
        if not family:
            return
        relatives = family.get("relatives", [])

        for relative in relatives:
            rel_id = relative.get("id", "Unknown")
            sex = relative.get("sex", "Unknown")
            vital_status = relative.get("vitalStatus", {}).get("status", "Unknown")
            age_at_death = self._parse_iso_duration(
                relative.get("vitalStatus", {})
                .get("ageAtDeath", {})
                .get("iso8601duration", "")
            )

            rel_phens = []
            for pf in relative.get("phenotypicFeatures", []):
                label = self._get_ontology_label(
                    pf.get("type", {}).get("label", "Unknown")
                )
                onset = self._parse_temporal(pf.get("onset"))
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

        # Optional: include pedigree structure
        pedigree = family.get("pedigree", {})
        if pedigree.get("persons"):
            summary = f"Pedigree defined with {len(pedigree['persons'])} individuals."
            self._add_medical_sentence("FAMILY_PEDIGREE", summary)


class ClinicalSummarizer:
    """LLM-powered clinical summary generator using a professional chat template."""

    def __init__(
        self, model=None, tokenizer=None, model_name: Optional[str] = None
    ) -> None:
        if model is not None:
            if tokenizer is None:
                raise ValueError(
                    "A tokenizer must be provided if a model instance is given."
                )
            self.pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        elif model_name is not None:
            self.pipeline = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=AutoTokenizer.from_pretrained(model_name),
                torch_dtype=torch.float16,
                device_map="auto",
            )
        else:
            raise ValueError(
                "Must provide either a model instance with its tokenizer or a model_name."
            )

    def generate_summary(self, sentences: List[str]) -> Dict:
        """Generate a structured clinical summary using an improved chat template."""
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
            if self.pipeline.tokenizer is None:
                raise ValueError("Tokenizer is not initialized.")

            prompt = self.pipeline.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            raw_outputs = self.pipeline(
                prompt,
                max_new_tokens=2048,
                do_sample=False,
                return_full_text=False,
            )

            # Convert generator/PipelineIterator/etc. to list
            if raw_outputs is None or not hasattr(raw_outputs, "__iter__"):
                raise ValueError("Pipeline output is not iterable.")

            outputs = list(raw_outputs)

            if not outputs or not isinstance(outputs[0], dict):
                raise ValueError("Invalid output format from pipeline.")

            generated_text = outputs[0].get("generated_text", "")
            return self._extract_llm_output(generated_text)

        except Exception as e:
            logging.error(f"LLM processing failed: {e}")
            return {
                "main_conditions": [],
                "other_conditions": [],
                "expanded_sentences": [],
                "error": str(e),
            }

    def _extract_llm_output(self, generated_text: str) -> Dict:
        """Robust JSON extraction with validation."""
        try:
            match = re.search(r"\{.*\}", generated_text, re.DOTALL)
            if not match:
                logging.error(
                    "No JSON object found in the LLM output: " + generated_text
                )
                return {
                    "error": "No JSON object found in LLM output",
                    "main_conditions": [],
                    "other_conditions": [],
                    "expanded_sentences": [],
                }

            json_str = match.group()
            # Remove markdown formatting if present
            json_str = re.sub(r"```(?:json)?\s*", "", json_str)
            json_str = re.sub(r"\s*```", "", json_str)

            result = json.loads(json_str)

            expected_keys = [
                "main_conditions",
                "other_conditions",
                "expanded_sentences",
            ]
            if not all(key in result for key in expected_keys):
                raise ValueError("Missing required JSON keys in LLM output")

            return result

        except json.JSONDecodeError as e:
            logging.error(f"JSON decoding failed: {e}")
            return {
                "error": f"Invalid JSON format: {str(e)}",
                "main_conditions": [],
                "other_conditions": [],
                "expanded_sentences": [],
            }
        except Exception as e:
            logging.error(f"Output parsing failed: {e}")
            return {
                "error": str(e),
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
    """End-to-end processing pipeline for Phenopacket data."""
    try:
        # Extract medical data from the Phenopacket
        processor = PhenopacketProcessor(input_file)
        narrative = processor.generate_medical_narrative()

        # Generate the clinical summary using the LLM.
        # Use the provided model instance (with tokenizer) if available.
        if model is not None and tokenizer is not None:
            summarizer = ClinicalSummarizer(model=model, tokenizer=tokenizer)
        else:
            summarizer = ClinicalSummarizer(model_name=model_name)

        summary = summarizer.generate_summary(narrative)

        # Save results to the output file
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)

        logging.info(
            f"Successfully processed {input_file} and saved results to {output_file}"
        )
        return True

    except Exception as e:
        logging.error(f"Processing failed: {e}")
        return False
