import logging

from trialmatchai.utils.logging_config import reset_request_id, set_request_id, setup_logging


class _CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records = []

    def emit(self, record) -> None:
        self.records.append(record)


def test_request_id_attached():
    logger = setup_logging(__name__)
    handler = _CaptureHandler()
    logger.addHandler(handler)

    token = set_request_id("patient-xyz")
    try:
        logger.info("hello")
    finally:
        reset_request_id(token)

    assert handler.records
    assert getattr(handler.records[0], "request_id", None) == "patient-xyz"
