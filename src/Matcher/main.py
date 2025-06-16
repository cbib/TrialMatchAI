#!/usr/bin/env python3
import socket
import subprocess
import time
import os
import json
import logging
import torch
import multiprocessing as mp
from peft.peft_model import PeftModel
from transformers import PreTrainedTokenizer
from typing import Tuple, cast

from elasticsearch import Elasticsearch
from transformers import (
    pipeline,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from ..Parser.biomedner_engine import BioMedNER

from . import parse_and_augment as pag
from . import first_level as fl
from . import second_level as sl
from . import cot_reasoning as cr
from . import trial_ranker as tr
from .llm_reranker import LLMReranker

# -------------------------
# Load configuration file
# -------------------------
CONFIG_PATH = "config.json"


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


config = load_config(CONFIG_PATH)

# Extract configuration sections for easier reference.
bio_med_ner_config = config["bio_med_ner"]
paths_config = config["paths"]
model_config = config["model"]
tokenizer_config = config["tokenizer"]
global_config = config["global"]
es_config = config["elasticsearch"]
first_level_embedder_config = config["first_level_embedder"]
retrieval_embedder_config = config["retrieval_embedder"]
rag_config = config["rag"]
llm_reranker_config = config["LLM_reranker"]
search_config = config["search"]

# ------------------------------------------------------------------
# BioMedNER Service Startup/Shutdown Functionality
# ------------------------------------------------------------------

bio_med_ner_params = {
    "biomedner_port": bio_med_ner_config["biomedner_port"],
    "gner_port": bio_med_ner_config["gner_port"],
    "gene_norm_port": bio_med_ner_config["gene_norm_port"],
    "disease_norm_port": bio_med_ner_config["disease_norm_port"],
}

RUN_SCRIPT = bio_med_ner_config["run_script"]
STOP_SCRIPT = bio_med_ner_config["stop_script"]


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


def check_ports_in_use(ports: list) -> bool:
    return any(is_port_in_use(port) for port in ports)


def run_script(script_path: str):
    print(f"Executing: {script_path}")
    subprocess.run(["bash", script_path], check=True)


def initialize_biomedner_services():
    ports_to_check = list(bio_med_ner_params.values())
    if check_ports_in_use(ports_to_check):
        print("Detected active services. Stopping running instances...")
        run_script(STOP_SCRIPT)
        print("Waiting for 10 seconds before restarting...")
        time.sleep(10)
    print("Starting BioMedNER services...")
    run_script(RUN_SCRIPT)
    print("BioMedNER services started successfully.")


# ------------------------------------------------------------------
# Global BioMedNER Parameters for downstream modules.
# ------------------------------------------------------------------

BIOMEDNER_PARAMS = {
    "biomedner_home": bio_med_ner_config["biomedner_home"],
    "biomedner_port": bio_med_ner_config["biomedner_port"],
    "gner_port": bio_med_ner_config["gner_port"],
    "gene_norm_port": bio_med_ner_config["gene_norm_port"],
    "disease_norm_port": bio_med_ner_config["disease_norm_port"],
    "use_neural_normalizer": bio_med_ner_config["use_neural_normalizer"],
    "no_cuda": bio_med_ner_config["no_cuda"],
}

##############################################
# Logging Configuration & Global Settings
##############################################
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Global paths derived from config.
PATIENTS_DIR = paths_config["patients_dir"]
OUTPUT_DIR = paths_config["output_dir"]

##############################################
# Global Variables for Model Sharing
##############################################
GLOBAL_MODEL = None
GLOBAL_TOKENIZER = None
GLOBAL_DEVICE = global_config["device"]  # Use GPU as defined in config


##############################################
# Top-level Function to Load the Model on a Single GPU
##############################################
def load_model_on_device(device):
    torch.cuda.set_device(device)
    logger.info(f"[Process PID={os.getpid()}] Loading model on GPU {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["base_model"],
        use_fast=tokenizer_config["use_fast"],
        padding_side=tokenizer_config["padding_side"],
    )
    # Convert compute dtype string to torch dtype.
    compute_dtype = (
        torch.float16
        if model_config["quantization"]["bnb_4bit_compute_dtype"].lower() == "float16"
        else torch.float32
    )
    quant_config = BitsAndBytesConfig(
        load_in_4bit=model_config["quantization"]["load_in_4bit"],
        bnb_4bit_use_double_quant=model_config["quantization"][
            "bnb_4bit_use_double_quant"
        ],
        bnb_4bit_quant_type=model_config["quantization"]["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_config["base_model"],
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=f"cuda:{device}",
        attn_implementation="flash_attention_2",
        quantization_config=quant_config,
    )
    logger.info(f"[Process PID={os.getpid()}] Model loaded on GPU {device}.")
    return (model, tokenizer)


def get_global_model_and_tokenizer():
    global GLOBAL_MODEL, GLOBAL_TOKENIZER
    ctx = mp.get_context("spawn")
    (model, tokenizer) = ctx.Pool(1).map(load_model_on_device, [GLOBAL_DEVICE])[0]
    GLOBAL_MODEL = model
    GLOBAL_TOKENIZER = tokenizer
    return model, tokenizer


def get_pipeline(model, tokenizer):
    return pipeline(
        "text-generation", model=model, tokenizer=tokenizer, torch_dtype=torch.float16
    )


def create_patient_output_folder(patient_id):
    folder = os.path.join(OUTPUT_DIR, patient_id)
    os.makedirs(folder, exist_ok=True)
    return folder


##############################################
# Monkey-Patching Function for cot_reasoning.BatchTrialProcessor
##############################################
def patch_cot_reasoning():
    from .cot_reasoning import BatchTrialProcessor

    def patched_init_model(self) -> Tuple[PeftModel, PreTrainedTokenizer]:
        global GLOBAL_MODEL, GLOBAL_TOKENIZER
        return cast(
            Tuple[PeftModel, PreTrainedTokenizer], (GLOBAL_MODEL, GLOBAL_TOKENIZER)
        )

    BatchTrialProcessor._init_model = patched_init_model
    logger.info(
        "Patched cot_reasoning.BatchTrialProcessor._init_model to reuse the global model."
    )


##############################################
# Step 1 & 2: Generate Keywords from Phenopacket
##############################################
def process_patient_keywords(phenopacket_path, output_folder, model, tokenizer):
    if not os.path.exists(phenopacket_path):
        logger.error(f"Phenopacket file does not exist: {phenopacket_path}")
        return None
    keywords_out_path = os.path.join(output_folder, "keywords.json")
    logger.info("Generating keywords using the Phenopacket processor ...")

    success = pag.process_phenopacket(
        phenopacket_path, keywords_out_path, model, tokenizer
    )
    if not success:
        logger.error(f"Phenopacket processing failed for {phenopacket_path}")
        return None
    with open(keywords_out_path, "r") as f:
        keywords = json.load(f)
    return keywords


##############################################
# Step 3: First-Level Trial Search
##############################################
def run_first_level(keywords, output_folder, patient_info, bio_med_ner, embedder):
    main_conditions = keywords.get("main_conditions", [])
    other_conditions = keywords.get("other_conditions", [])
    expanded_sentences = keywords.get("expanded_sentences", [])
    if not main_conditions:
        logger.error("No main_conditions found in keywords.")
        return None, None, None, None, None

    condition = main_conditions[0]
    age = patient_info.get("age", "all")
    sex = patient_info.get("gender", "all")
    overall_status = "All"

    es_client = Elasticsearch(
        hosts=[es_config["host"]],
        ca_certs=paths_config["docker_certs"],
        basic_auth=(es_config["username"], es_config["password"]),
        request_timeout=es_config["request_timeout"],
        retry_on_timeout=es_config["retry_on_timeout"],
    )

    index_name = es_config["index_trials_v1"]
    cts = fl.ClinicalTrialSearch(es_client, embedder, index_name, bio_med_ner)

    synonyms = cts.get_synonyms(condition.lower().strip())
    main_conditions.extend(synonyms[0:10])
    logger.info(f"Synonyms: {synonyms}")

    trials, scores = cts.search_trials(
        condition=condition,
        age_input=age,
        sex=sex,
        overall_status=overall_status,
        size=500,
        pre_selected_nct_ids=None,
        synonyms=main_conditions,
        other_conditions=other_conditions,
        vector_score_threshold=search_config["vector_score_threshold"],
    )
    nct_ids = [trial.get("nct_id") for trial in trials if trial.get("nct_id")]
    first_level_scores = {}
    for trial, score in zip(trials, scores):
        nid = trial.get("nct_id")
        if nid:
            first_level_scores[nid] = score

    nct_ids_path = os.path.join(output_folder, "nct_ids.txt")
    with open(nct_ids_path, "w") as f:
        for nid in nct_ids:
            f.write(str(nid) + "\n")
    first_level_scores_path = os.path.join(output_folder, "first_level_scores.json")
    with open(first_level_scores_path, "w") as f:
        json.dump(first_level_scores, f)
    logger.info(f"First-level search complete: {len(nct_ids)} trial IDs saved.")
    return (
        nct_ids,
        main_conditions,
        other_conditions,
        expanded_sentences,
        first_level_scores,
    )


##############################################
# Step 4: Second-Level Retrieval & Semi-Final Ranking
##############################################
def run_second_level(
    output_folder,
    nct_ids,
    main_conditions,
    other_conditions,
    expanded_sentences,
    gemma_retriever,
    first_level_scores,
):
    queries = list(set(main_conditions + other_conditions + expanded_sentences))
    logger.info(f"Running second-level retrieval with {len(queries)} queries ...")
    synonyms = gemma_retriever.get_synonyms(queries[0])
    queries.extend(synonyms)
    second_level_results = gemma_retriever.retrieve_and_rank(
        queries, nct_ids, top_n=len(nct_ids)
    )
    combined_scores = {}
    for trial in second_level_results:
        trial_id = trial["nct_id"]
        second_score = trial["score"]
        first_score = first_level_scores.get(trial_id, 0)
        combined_scores[trial_id] = first_score + second_score

    sorted_trials = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    num_top = max(1, len(sorted_trials) // 2)
    semi_final_trials = sorted_trials[:num_top]

    top_trials_path = os.path.join(output_folder, "top_trials.txt")
    with open(top_trials_path, "w") as f:
        for trial_id, avg_score in semi_final_trials:
            f.write(trial_id + "\n")
    logger.info("Second-level retrieval and ranking complete. Top trials saved.")
    return semi_final_trials, top_trials_path


##############################################
# Step 5: RAG-based Trial Matching
##############################################
def run_rag_global(output_folder, top_trials_file, patient_info):
    with open(top_trials_file, "r") as f:
        top_trials = [line.strip() for line in f if line.strip()]
    if not top_trials:
        logger.error("No top trials available for RAG processing.")
        return
    patient_profile = patient_info.get("split_raw_description", [])
    if not patient_profile:
        logger.error("No patient profile available for RAG processing.")
        return

    rag_processor = cr.BatchTrialProcessor(
        base_model=model_config[
            "base_model"
        ],  # Value now irrelevant due to monkey-patching.
        device=GLOBAL_DEVICE,
        batch_size=rag_config["batch_size"],
    )
    logger.info(
        f"Using global RAG processor on GPU {GLOBAL_DEVICE} for patient in folder {output_folder} ..."
    )
    rag_processor.process_trials(
        nct_ids=top_trials,
        json_folder=paths_config["trials_json_folder"],
        output_folder=output_folder,
        patient_profile=patient_profile,
    )
    rag_out_file = os.path.join(output_folder, "rag_output.json")
    with open(rag_out_file, "w") as f:
        json.dump({"status": "done"}, f)
    logger.info("RAG-based trial matching complete for patient.")


##############################################
# Step 6: Final Ranking
##############################################
def run_final_ranking(output_folder):
    ranked_trials_path = os.path.join(output_folder, "ranked_trials.json")
    trials = tr.load_trial_data(output_folder)
    ranked_trials = tr.rank_trials(trials)
    tr.save_ranked_trials(ranked_trials, ranked_trials_path)
    logger.info("Final ranking complete.")
    return ranked_trials_path


##############################################
# Main Pipeline
##############################################
def main_pipeline():
    logger.info("Starting integrated TrialMatchAI pipeline ...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load the global model and tokenizer.
    logger.info("Loading global model and tokenizer ...")
    base_model, base_tokenizer = get_global_model_and_tokenizer()

    # Apply monkey-patching so that BatchTrialProcessor reuses the global model.
    patch_cot_reasoning()
    logger.info("Global model loaded successfully.")

    # Load the fine-tuned adapter.
    base_model = PeftModel.from_pretrained(
        base_model, model_config["fine_tuned_adapter_phi"]
    )
    base_model = base_model.eval()
    logger.info("Fine-tuned model adapter loaded and set to eval mode.")

    # Load first-level embedder.
    logger.info("Loading first-level embedder ...")
    first_level_embedder = fl.QueryEmbedder(
        model_name=first_level_embedder_config["model_name"],
        max_length=first_level_embedder_config["max_length"],
    )

    # Load BioMedNER.
    bio_med_ner_instance = BioMedNER(**BIOMEDNER_PARAMS)

    # Load GEMMA model for LLM reranking.
    logger.info("Loading GEMMA model for LLM reranking ...")
    es_client_for_gemma = Elasticsearch(
        hosts=[es_config["host"]],
        ca_certs=paths_config["docker_certs"],
        basic_auth=(es_config["username"], es_config["password"]),
        request_timeout=es_config["request_timeout"],
        retry_on_timeout=es_config["retry_on_timeout"],
    )
    llm_reranker_model_path = model_config["reranker_model_path"]
    adapter_path = model_config["reranker_adapter_path"]
    llm_reranker_model = LLMReranker(
        model_path=llm_reranker_model_path,
        adapter_path=adapter_path,
        torch_dtype=torch.float16,
        batch_size=llm_reranker_config["batch_size"],
    )
    retrieval_embedder = sl.SecondLevelSentenceEmbedder(
        retrieval_embedder_config["model_name"]
    )

    index_name = es_config["index_trials_eligibility"]
    gemma_retriever = sl.SecondStageRetriever(
        es_client_for_gemma,
        llm_reranker_model,
        retrieval_embedder,
        index_name,
        bio_med_ner=bio_med_ner_instance,
    )

    # Process each patient subfolder in PATIENTS_DIR.
    patient_dirs = [
        d
        for d in os.listdir(PATIENTS_DIR)
        if os.path.isdir(os.path.join(PATIENTS_DIR, d))
    ]
    for patient_id in patient_dirs:
        logger.info(f"Processing patient {patient_id} ...")
        patient_input_folder = os.path.join(PATIENTS_DIR, patient_id)
        patient_output_folder = create_patient_output_folder(patient_id)

        # Skip if already processed.
        rank_file = os.path.join(patient_output_folder, "ranked_trials.json")
        if os.path.exists(rank_file):
            logger.info(f"Patient {patient_id} already fully processed. Skipping.")
            continue

        # The phenopacket is the sole patient information.
        phenopacket_path = os.path.join(patient_input_folder, "phenopacket.json")
        if not os.path.exists(phenopacket_path):
            logger.error(
                f"Phenopacket file missing for patient {patient_id}. Skipping."
            )
            continue

        # Load the phenopacket to use as patient_info for downstream steps.
        try:
            with open(phenopacket_path, "r") as f:
                patient_info = json.load(f)
        except Exception as e:
            logger.error(f"Error loading phenopacket for patient {patient_id}: {e}")
            continue

        # Step 1 & 2: Generate keywords using the Phenopacket processor.
        keywords_file = os.path.join(patient_output_folder, "keywords.json")
        if not os.path.exists(keywords_file):
            keywords = process_patient_keywords(
                phenopacket_path, patient_output_folder, base_model, base_tokenizer
            )
            if keywords is None:
                logger.error(
                    f"Skipping patient {patient_id} due to keyword generation failure."
                )
                continue
        else:
            with open(keywords_file, "r") as f:
                keywords = json.load(f)
        # Use the expanded sentences from the keywords as the patient profile.
        patient_info["split_raw_description"] = keywords.get("expanded_sentences", [])

        # Step 3: First-level trial search.
        nct_ids_file = os.path.join(patient_output_folder, "nct_ids.txt")
        first_level_scores_file = os.path.join(
            patient_output_folder, "first_level_scores.json"
        )
        if not os.path.exists(nct_ids_file) or not os.path.exists(
            first_level_scores_file
        ):
            result = run_first_level(
                keywords,
                patient_output_folder,
                patient_info,
                bio_med_ner_instance,
                first_level_embedder,
            )
            if result is None:
                logger.error(
                    f"Skipping patient {patient_id} due to first-level search failure."
                )
                continue
            (
                nct_ids,
                main_conditions,
                other_conditions,
                expanded_sentences,
                first_level_scores,
            ) = result
        else:
            with open(nct_ids_file, "r") as f:
                nct_ids = [line.strip() for line in f if line.strip()]
            with open(first_level_scores_file, "r") as f:
                first_level_scores = json.load(f)
            main_conditions = keywords.get("main_conditions", [])
            other_conditions = keywords.get("other_conditions", [])
            expanded_sentences = keywords.get(
                "expanded_sentences", []
            ) + patient_info.get("expanded_sentences", [])
        if not nct_ids:
            logger.error(
                f"No trial IDs retrieved for patient {patient_id}. Skipping further steps."
            )
            continue

        # Step 4: Second-level retrieval & semi-final ranking.
        top_trials_file = os.path.join(patient_output_folder, "top_trials.txt")
        if not os.path.exists(top_trials_file):
            semi_final_trials, top_trials_file = run_second_level(
                patient_output_folder,
                nct_ids,
                main_conditions,
                other_conditions,
                expanded_sentences,
                gemma_retriever,
                first_level_scores,
            )

        # Step 5: RAG-based trial matching.
        rag_out_file = os.path.join(patient_output_folder, "rag_output.json")
        if not os.path.exists(rag_out_file):
            run_rag_global(patient_output_folder, top_trials_file, patient_info)

        # Step 6: Final ranking.
        ranked_file = os.path.join(patient_output_folder, "ranked_trials.json")
        if not os.path.exists(ranked_file):
            run_final_ranking(patient_output_folder)

    logger.info("Integrated TrialMatchAI pipeline completed successfully.")


##############################################
# Entry Point
##############################################
if __name__ == "__main__":
    initialize_biomedner_services()
    main_pipeline()
