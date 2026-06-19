import socket
import subprocess
import time

from Matcher.utils.logging_config import setup_logging

logger = setup_logging(__name__)


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


def check_ports_in_use(ports: list) -> bool:
    return any(is_port_in_use(port) for port in ports)


def run_script(script_path: str):
    logger.info(f"Executing: {script_path}")
    subprocess.run(["bash", script_path], check=True)


def initialize_biomedner_services(config: dict):
    if not config.get("services", {}).get("auto_start", False):
        logger.info(
            "BioMedNER auto-start is disabled. Set TRIALMATCHAI_BIOMEDNER_AUTO_START=true to start local services."
        )
        return

    ports_to_check = [
        config["bio_med_ner"]["biomedner_port"],
        config["bio_med_ner"]["gner_port"],
        config["bio_med_ner"]["gene_norm_port"],
        config["bio_med_ner"]["disease_norm_port"],
    ]
    if check_ports_in_use(ports_to_check):
        logger.info("Detected active services. Stopping running instances...")
        run_script(config["services"]["stop_script"])
        logger.info("Waiting for 10 seconds before restarting...")
        time.sleep(10)
    logger.info("Starting BioMedNER services...")
    run_script(config["services"]["run_script"])
    logger.info("BioMedNER services started successfully.")
