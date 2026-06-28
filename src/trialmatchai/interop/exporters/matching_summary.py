from __future__ import annotations

from trialmatchai.interop.models import PatientProfile
from trialmatchai.interop.narrative import render_patient_narrative, render_search_terms


def profile_to_matching_summary(profile: PatientProfile) -> dict:
    main_conditions, other_conditions = render_search_terms(profile)
    age = (
        int(profile.demographics.age_years)
        if profile.demographics.age_years is not None
        else "all"
    )
    gender = profile.demographics.sex or profile.demographics.gender or "all"
    patient_narrative = render_patient_narrative(profile)
    return {
        "patient_id": profile.patient_id,
        "main_conditions": main_conditions,
        "other_conditions": other_conditions,
        "patient_narrative": patient_narrative,
        "age": age,
        "gender": gender,
        "provenance": [
            provenance.model_dump(mode="json", exclude_none=True)
            for provenance in profile.provenance
        ],
    }
