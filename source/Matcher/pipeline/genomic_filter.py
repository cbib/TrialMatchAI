from typing import Dict, List

from Matcher.utils.logging_config import setup_logging

logger = setup_logging(__name__)


def extract_genomic_terms(phenopacket: Dict) -> List[str]:
    """
    Extracts search-relevant terms from phenopacket genomicInterpretations.

    Returns a deduplicated list of strings (gene symbols, HGVS labels, amino
    acid changes, therapeutic actionability labels, and the disease label from
    each interpretation) suitable for injection into keyword search queries.
    """
    terms: set = set()

    for interp in phenopacket.get("interpretations", []):
        dx = interp.get("diagnosis", {})

        disease_label = dx.get("disease", {}).get("label", "")
        if disease_label:
            terms.add(disease_label)

        for gi in dx.get("genomicInterpretations", []):
            vi = gi.get("variantInterpretation", {})
            vd = vi.get("variationDescriptor", {})

            gene = vd.get("geneContext", {}).get("symbol", "")
            if gene:
                terms.add(gene)

            hgvs_label = vd.get("label", "")
            if hgvs_label:
                terms.add(hgvs_label)

            aa_change = (
                vd.get("molecularAttributes", {})
                .get("aminoAcidChange", {})
                .get("value", "")
            )
            if aa_change:
                terms.add(aa_change)

            ta_label = vi.get("therapeuticActionability", {}).get("label", "")
            if ta_label:
                terms.add(ta_label)

    result = sorted(terms)
    if result:
        logger.debug("Genomic terms extracted: %s", result)
    return result
