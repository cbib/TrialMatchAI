"""Optional geographic (country-level, site-aware) trial filtering.

Trials recruit at many sites across countries, so the filter is deliberately
recall-safe: a trial passes when the patient's country is unknown, when the
trial has no indexed site countries, or when ANY of its sites is in the
patient's country. It only drops trials we are confident have no site in the
patient's country. Country granularity avoids the recall risk of distance/region
matching on messy ClinicalTrials.gov site data.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def patient_country(profile: Any) -> str | None:
    location = getattr(profile, "location", None)
    country = getattr(location, "country", None) if location is not None else None
    country = (country or "").strip()
    return country or None


def trial_site_countries(trial: Mapping[str, Any]) -> set[str]:
    countries: set[str] = set()
    locations = trial.get("location")
    if isinstance(locations, Mapping):
        locations = [locations]
    if not isinstance(locations, Sequence) or isinstance(locations, (str, bytes)):
        return countries
    for site in locations:
        if isinstance(site, Mapping):
            country = str(site.get("country") or "").strip().casefold()
            if country:
                countries.add(country)
    return countries


def trial_in_country(trial: Mapping[str, Any], country: str) -> bool:
    """True if the trial should pass the country filter (recall-safe)."""
    site_countries = trial_site_countries(trial)
    if not site_countries:
        return True  # unknown location -> do not drop
    return country.strip().casefold() in site_countries


def filter_trials_by_country(
    trials: list[dict[str, Any]],
    scores: list[float],
    country: str | None,
) -> tuple[list[dict[str, Any]], list[float]]:
    """Drop trials with known sites that exclude the patient's country."""
    if not country:
        return trials, scores
    kept_trials: list[dict[str, Any]] = []
    kept_scores: list[float] = []
    for trial, score in zip(trials, scores):
        if trial_in_country(trial, country):
            kept_trials.append(trial)
            kept_scores.append(score)
    return kept_trials, kept_scores
