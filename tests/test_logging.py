import logging
import unittest

from Matcher.utils.logging_config import reset_request_id, set_request_id, setup_logging


class CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records = []

    def emit(self, record) -> None:
        self.records.append(record)


class TestLoggingContext(unittest.TestCase):
    def test_request_id_is_attached(self) -> None:
        logger = setup_logging(__name__)
        handler = CaptureHandler()
        logger.addHandler(handler)

        token = set_request_id("patient-123")
        try:
            logger.info("hello")
        finally:
            reset_request_id(token)

        self.assertTrue(handler.records)
        record = handler.records[0]
        self.assertEqual(getattr(record, "request_id", None), "patient-123")


if __name__ == "__main__":
    unittest.main()
