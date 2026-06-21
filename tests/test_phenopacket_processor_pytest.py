import json

from trialmatchai.matching.phenopacket_processor import PhenopacketProcessor


def test_phenopacket_processor_minimal(tmp_path):
    data = {"id": "patient-1", "metaData": {}, "subject": {}}
    path = tmp_path / "patient.json"
    path.write_text(json.dumps(data))

    processor = PhenopacketProcessor(str(path))
    narrative = processor.generate_medical_narrative()
    assert narrative
