from __future__ import annotations

DEFAULT_REGISTRY_KEYWORDS: tuple[str, ...] = (
    "cancer OR oncology OR neoplasm OR tumor",
    "heart failure OR coronary artery disease OR atrial fibrillation",
    "stroke OR Parkinson disease OR Alzheimer disease OR epilepsy",
    "rare disease OR orphan disease",
    "autoimmune disease OR immunotherapy OR inflammatory disease",
    "COVID-19 OR HIV OR tuberculosis OR infectious disease",
    "diabetes OR obesity OR metabolic syndrome",
    "leukemia OR lymphoma OR anemia OR hematology",
    "pediatric OR children OR adolescent",
    "precision medicine OR genomic OR biomarker",
)

DEFAULT_REGISTRY_STATUSES: tuple[str, ...] = (
    "RECRUITING",
    "NOT_YET_RECRUITING",
    "ACTIVE_NOT_RECRUITING",
)
