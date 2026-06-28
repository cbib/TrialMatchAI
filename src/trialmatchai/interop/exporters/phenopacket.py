from __future__ import annotations

from trialmatchai.interop.models import ClinicalFact, PatientProfile


def profile_to_phenopacket(profile: PatientProfile) -> dict:
    unsupported: list[dict] = []
    packet = {
        "id": profile.patient_id,
        "subject": {
            "id": profile.patient_id,
            **({"sex": profile.demographics.sex.upper()} if profile.demographics.sex else {}),
            **(
                {"dateOfBirth": profile.demographics.birth_date.isoformat()}
                if profile.demographics.birth_date
                else {}
            ),
        },
        "phenotypicFeatures": [_phenotype_fact(fact) for fact in profile.phenotypes],
        "diseases": [_disease_fact(fact) for fact in profile.conditions],
        "measurements": [_measurement_fact(fact) for fact in profile.observations],
        "medicalActions": [
            *[_treatment_fact(fact) for fact in profile.medications],
            *[_procedure_fact(fact) for fact in profile.procedures],
        ],
        "metaData": {
            "createdBy": "TrialMatchAI",
            "resources": [],
        },
    }
    for fact in [
        *profile.diagnostic_reports,
        *profile.genomic_findings,
        *profile.cancer_profile,
        *profile.family_history,
    ]:
        unsupported.append(
            {
                "fact_id": fact.fact_id,
                "category": fact.category,
                "label": fact.label,
                "reason": "No lossless Phenopacket v1 exporter mapping implemented.",
            }
        )
    packet["trialmatchaiConversionReport"] = {
        "lossy": bool(unsupported),
        "unsupported": unsupported,
    }
    return packet


def _ontology_class(fact: ClinicalFact) -> dict:
    code = fact.normalized_codes[0] if fact.normalized_codes else fact.original_code
    if code is None:
        return {"id": fact.label, "label": fact.label}
    return {"id": f"{code.vocabulary}:{code.code}", "label": code.label or fact.label}


def _phenotype_fact(fact: ClinicalFact) -> dict:
    return {
        "type": _ontology_class(fact),
        "excluded": fact.negated,
        **({"description": fact.description} if fact.description else {}),
    }


def _disease_fact(fact: ClinicalFact) -> dict:
    return {
        "term": _ontology_class(fact),
        **({"excluded": fact.negated} if fact.negated else {}),
        **({"description": fact.description} if fact.description else {}),
    }


def _measurement_fact(fact: ClinicalFact) -> dict:
    return {
        "assay": _ontology_class(fact),
        "value": {"value": fact.description or fact.label},
    }


def _treatment_fact(fact: ClinicalFact) -> dict:
    return {"treatment": {"agent": _ontology_class(fact)}}


def _procedure_fact(fact: ClinicalFact) -> dict:
    return {"procedure": {"code": _ontology_class(fact)}}
