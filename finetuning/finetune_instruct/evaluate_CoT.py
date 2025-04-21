import torch
import json
from bert_score import score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel
import random

def load_model(device, model_path, adapter_path):
    """
    Loads the model on the specified device using 4-bit quantization.
    """
    print(f"Loading model on device cuda:{device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=f"cuda:{device}",
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        quantization_config=quant_config
    )

    pipe = pipeline(
        "text-generation",
        model=base_model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        do_sample=False,
        repetition_penalty=1.05
    )

    # Load the fine-tuned adapter model
    pipe.model = PeftModel.from_pretrained(base_model, adapter_path)

    return pipe, tokenizer

def generate_output(pipe, tokenizer, instruction, input_text):
    """
    Generates model output using the instruction as system prompt and input as user prompt.
    This version instructs the model to include its internal chain-of-thought exactly once,
    followed by the final answer.
    """
    # Instruct the model to reveal its chain-of-thought once.
    system_msg = (instruction +
                  "\nPlease include your internal chain-of-thought exactly once, "
                  "followed by the final answer. Do not repeat the chain-of-thought.")
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": input_text}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    generated = pipe(prompt)[0]['generated_text'].strip()

    # Optional post-processing: If the chain-of-thought is repeated, keep only the first occurrence.
    # This snippet assumes the model labels its reasoning with "Chain-of-thought:" and final answer with "Final Answer:".
    if generated.count("Chain-of-thought:") > 1:
        # Split on the label and reconstruct output using only the first instance.
        parts = generated.split("Chain-of-thought:")
        first_cot = parts[1].split("Final Answer:")[0] if "Final Answer:" in parts[1] else parts[1]
        final_answer = ""
        if "Final Answer:" in generated:
            final_answer = "Final Answer:" + generated.split("Final Answer:")[-1]
        # Rebuild the output with a single chain-of-thought section.
        generated = parts[0] + "Chain-of-thought:" + first_cot + "\n" + final_answer
        generated = generated.strip()

    return generated

def compute_bertscore(model_outputs, reference_outputs, lang='en', model_type='allenai/longformer-base-4096'):
    """
    Computes BERTScore for evaluating model-generated outputs against reference texts.
    """
    assert len(model_outputs) == len(reference_outputs), "Mismatch in number of model and reference outputs"

    print("Computing BERTScore...")
    precision, recall, f1 = score(
        model_outputs,
        reference_outputs,
        lang=lang,
        model_type=model_type,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    return {
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "f1": f1.mean().item()
    }

def main():
    device = 0
    model_path = "microsoft/phi-4"
    adapter_path = "finetuned_phi_reasoning/"
    pipe, tokenizer = load_model(device, model_path, adapter_path)

    file_path = "finetuning_data/medical_o1_reasoning_test.jsonl"
    model_outputs = []
    reference_outputs = []

    with open(file_path, "r") as f:
        lines = f.readlines()

    # Randomly select 15 test cases
    selected_lines = random.sample(lines, 500)

    for idx, line in enumerate(selected_lines, 1):
        print(f"Processing randomly selected line {idx}")
        data = json.loads(line)
        generated_text = generate_output(pipe, tokenizer, data["instruction"], data["input"])
        print("Generated Output:\n", generated_text, "\n")
        model_outputs.append(generated_text)
        reference_outputs.append(data["output"])

    results = compute_bertscore(model_outputs, reference_outputs)

    print("\nBERTScore Results:")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")

    with open("bertscore_results.txt", "w") as f:
        f.write(f"Precision: {results['precision']:.4f}\n")
        f.write(f"Recall: {results['recall']:.4f}\n")
        f.write(f"F1 Score: {results['f1']:.4f}\n")

if __name__ == "__main__":
    main()
