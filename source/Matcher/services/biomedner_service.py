import socket
import subprocess
import time

from Matcher.utils.logging_config import setup_logging

logger = setup_logging(__name__)

BIOMEDNER_STARTUP_TIMEOUT = 300  # seconds — RoBERTa NER needs ~130s to load on cold start


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


def check_ports_in_use(ports: list) -> bool:
    return any(is_port_in_use(port) for port in ports)


def wait_for_ports(ports: list, timeout: int = BIOMEDNER_STARTUP_TIMEOUT) -> bool:
    logger.info("Waiting for BioMedNER services on ports %s (timeout: %ds)...", ports, timeout)
    deadline = time.time() + timeout
    while time.time() < deadline:
        ready = [p for p in ports if is_port_in_use(p)]
        if len(ready) == len(ports):
            logger.info("All BioMedNER services ready.")
            return True
        logger.info("Waiting... ports ready: %s / %s", len(ready), len(ports))
        time.sleep(5)
    logger.warning(
        "BioMedNER startup timed out after %ds. Only %d/%d ports ready. "
        "Synonym expansion may be degraded.",
        timeout,
        len([p for p in ports if is_port_in_use(p)]),
        len(ports),
    )
    return False


def run_script(script_path: str):
    logger.info(f"Executing: {script_path}")
    subprocess.run(["bash", script_path], check=True)


def initialize_biomedner_services(config: dict):
    ports_to_check = [
        config["bio_med_ner"]["biomedner_port"],
        config["bio_med_ner"]["gner_port"],
        config["bio_med_ner"]["gene_norm_port"],
        config["bio_med_ner"]["disease_norm_port"],
    ]
    if all(is_port_in_use(p) for p in ports_to_check):
        logger.info("BioMedNER services already running on all ports. Skipping startup.")
        return
    if check_ports_in_use(ports_to_check):
        logger.info("Detected partial services. Stopping running instances...")
        run_script(config["services"]["stop_script"])
        logger.info("Waiting for 10 seconds before restarting...")
        time.sleep(10)
    logger.info("Starting BioMedNER services...")
    run_script(config["services"]["run_script"])
    wait_for_ports(ports_to_check)
