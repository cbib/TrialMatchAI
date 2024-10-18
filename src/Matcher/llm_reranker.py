import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Tuple, Optional, Dict
import gc

class Reranker:
    def __init__(
        self,
        base_model: str,
        adapter_path: str,
        prompt: Optional[str] = None,
        max_length: int = 1024,
        use_fp16: bool = False  # Flag to enable mixed precision
    ):
        """
        Initializes the SimpleReranker with the specified base model, adapter, and tokenizer.

        Args:
            base_model (str): The name or path of the pretrained base model.
            adapter_path (str): The path to the local PEFT adapter directory.
            prompt (Optional[str]): The prompt to use for analysis. If None, a default prompt is used.
            max_length (int): The maximum length for tokenization.
            use_fp16 (bool): Whether to use mixed precision (FP16) for inference.

        Raises:
            ValueError: If the base model or adapter cannot be loaded.
        """
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.max_length = max_length
        self.use_fp16 = use_fp16

        # Initialize the tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        except Exception as e:
            raise ValueError(f"Failed to load tokenizer for base model '{self.base_model}': {e}")

        # Initialize the base model
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.base_model)
        except Exception as e:
            raise ValueError(f"Failed to load base model '{self.base_model}': {e}")

        # Load the PEFT adapter
        try:
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
        except Exception as e:
            raise ValueError(f"Failed to load PEFT adapter from '{self.adapter_path}': {e}")

        self.model.eval()

        # Move model to appropriate device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        print(f"Model is using device: {self.device}")

        # Define the prompt
        self.prompt = (
            "Analyze the medical condition described in query A and the statement in passage B. "
            "Determine if the statement is in accordance with the condition. "
            "Respond with 'Yes' if they are suitable and aligned or 'No' if otherwise."
        ) if prompt is None else prompt

        # Precompute token IDs for efficiency
        self.sep_token = "\n"
        self.bos_token_id = self.tokenizer.bos_token_id
        yes_token_ids = self.tokenizer('Yes', add_special_tokens=False)['input_ids']
        if not yes_token_ids:
            raise ValueError("The tokenizer could not find the token ID for 'Yes'.")
        self.yes_token_id = yes_token_ids[0]

    def get_input(
        self,
        pair: Tuple[str, str]
    ) -> Dict[str, torch.Tensor]:
        """
        Prepares the input tensor for the model based on the input pair.

        Args:
            pair (Tuple[str, str]): A tuple containing a query and a passage.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the input_ids and attention_mask tensors.
        """
        query, passage = pair
        sep = self.sep_token

        # Tokenize prompt and separator
        prompt_inputs = self.tokenizer(
            self.prompt,
            return_tensors=None,
            add_special_tokens=False
        )['input_ids']
        sep_inputs = self.tokenizer(
            sep,
            return_tensors=None,
            add_special_tokens=False
        )['input_ids']

        # Tokenize query and passage
        query_inputs = self.tokenizer(
            f'A: {query}',
            return_tensors=None,
            add_special_tokens=False,
            max_length=self.max_length * 3 // 4,
            truncation=True
        )
        passage_inputs = self.tokenizer(
            f'B: {passage}',
            return_tensors=None,
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True
        )

        # Prepare combined input
        input_ids = [self.bos_token_id] + query_inputs['input_ids'] + sep_inputs + passage_inputs['input_ids'] + prompt_inputs
        attention_mask = [1] * len(input_ids)

        # Pad to max_length
        padding_length = self.max_length + len(sep_inputs) + len(prompt_inputs) - len(input_ids)
        if padding_length > 0:
            input_ids += [self.tokenizer.pad_token_id] * padding_length
            attention_mask += [0] * padding_length

        input_ids_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        attention_mask_tensor = torch.tensor([attention_mask], dtype=torch.long).to(self.device)

        return {
            'input_ids': input_ids_tensor,
            'attention_mask': attention_mask_tensor
        }

    def score_pair(
        self,
        pair: Tuple[str, str]
    ) -> float:
        """
        Computes the score for a single pair indicating the alignment based on the model's logits.

        Args:
            pair (Tuple[str, str]): A tuple containing a query and a passage.

        Returns:
            float: The score indicating the alignment for the pair.
        """
        with torch.no_grad():
            inputs = self.get_input(pair)
            if self.use_fp16 and self.device.startswith('cuda'):
                with torch.cuda.amp.autocast():
                    outputs = self.model(**inputs, return_dict=True)
            else:
                outputs = self.model(**inputs, return_dict=True)
            logits = outputs.logits  # Shape: (1, sequence_length, vocab_size)
            # Extract logits for the last token
            last_token_logits = logits[:, -1, :]  # Shape: (1, vocab_size)
            # Get the logits corresponding to the 'Yes' token
            yes_logits = last_token_logits[:, self.yes_token_id]
            return yes_logits.item()

    def compute_score(
        self,
        pair: Tuple[str, str]
    ) -> float:
        """
        Analyzes the given pair and returns its corresponding score.

        Args:
            pair (Tuple[str, str]): A tuple containing a query and a passage.

        Returns:
            float: The score indicating the alignment for the pair.
        """
        score = self.score_pair(pair)
        # Optional: Garbage collection to free up memory
        gc.collect()
        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()
        return score

# Example usage:
if __name__ == "__main__":
    base_model = "gpt-2"  # Replace with your base model
    adapter_path = "./adapter"  # Replace with your adapter path

    reranker = SimpleReranker(
        base_model=base_model,
        adapter_path=adapter_path,
        max_length=1024,
        use_fp16=False
    )

    query = "Patient experiences severe headaches."
    passage = "Headaches can be a symptom of various medical conditions."

    score = reranker.compute_score((query, passage))
    print(f"Alignment score: {score}")


# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel
# import multiprocessing as mp


# class Reranker:
#     def __init__(self, model_name, adapter_path, device_ids=None, batch_size=16):
#         """
#         Initialize the Reranker class with the model name, adapter path, device IDs, and batch size.
        
#         Args:
#         - model_name: The model name or path for the base model.
#         - adapter_path: The path to the fine-tuned adapter.
#         - device_ids: List of device IDs (GPUs) to use for parallel processing.
#         - batch_size: Batch size for processing.
#         """
#         self.model_name = model_name
#         self.adapter_path = adapter_path
#         self.device_ids = device_ids if device_ids is not None else list(range(torch.cuda.device_count()))
#         self.batch_size = batch_size

#     @staticmethod
#     def get_inputs(pairs, tokenizer, prompt=None, max_length=1024):
#         """
#         Prepare input tokens for the model based on the pairs (query, passage) and a prompt.
#         """
#         if prompt is None:
#             prompt = "Analyze the medical condition described in query A and the statement in passage B. Determine if the statement is in accordance with the condition. Respond with 'Yes' if they are suitable and aligned or 'No' if otherwise."
        
#         sep = "\n"
#         prompt_inputs = tokenizer(prompt, return_tensors=None, add_special_tokens=False)['input_ids']
#         sep_inputs = tokenizer(sep, return_tensors=None, add_special_tokens=False)['input_ids']
#         inputs = []
#         for query, passage in pairs:
#             query_inputs = tokenizer(f'A: {query}', return_tensors=None, add_special_tokens=False, max_length=max_length * 3 // 4, truncation=True)
#             passage_inputs = tokenizer(f'B: {passage}', return_tensors=None, add_special_tokens=False, max_length=max_length, truncation=True)
#             item = tokenizer.prepare_for_model(
#                 [tokenizer.bos_token_id] + query_inputs['input_ids'],
#                 sep_inputs + passage_inputs['input_ids'],
#                 truncation='only_second',
#                 max_length=max_length,
#                 padding=False,
#                 return_attention_mask=False,
#                 return_token_type_ids=False,
#                 add_special_tokens=False
#             )
#             item['input_ids'] = item['input_ids'] + sep_inputs + prompt_inputs
#             item['attention_mask'] = [1] * len(item['input_ids'])
#             inputs.append(item)
#         return tokenizer.pad(
#             inputs,
#             padding=True,
#             max_length=max_length + len(sep_inputs) + len(prompt_inputs),
#             pad_to_multiple_of=8,
#             return_tensors='pt',
#         )

#     def process_on_gpu(self, pairs, device_id):
#         """
#         Process a batch of data on a specific GPU.
#         """
#         # Load the tokenizer and model on the specific GPU
#         tokenizer = AutoTokenizer.from_pretrained(self.model_name)
#         model = AutoModelForCausalLM.from_pretrained(self.model_name).to(f'cuda:{device_id}')
#         model = PeftModel.from_pretrained(model, self.adapter_path).to(f'cuda:{device_id}')
#         model.eval()

#         yes_loc = tokenizer('Yes', add_special_tokens=False)['input_ids'][0]

#         # Process data in batches
#         all_scores = []
#         for i in range(0, len(pairs), self.batch_size):
#             batch_pairs = pairs[i:i + self.batch_size]
#             with torch.no_grad():
#                 inputs = self.get_inputs(batch_pairs, tokenizer).to(f'cuda:{device_id}')
#                 logits = model(**inputs, return_dict=True).logits[:, -1, yes_loc].view(-1).float()
#                 all_scores.extend(logits.cpu().tolist())

#         return all_scores

#     def compute_score(self, pairs):
#         """
#         Distribute data across the specified GPUs for parallel processing.
#         """
#         # Split the pairs into chunks for each GPU
#         num_gpus = len(self.device_ids)
#         chunk_size = len(pairs) // num_gpus
#         pair_chunks = [pairs[i * chunk_size:(i + 1) * chunk_size] for i in range(num_gpus)]

#         # Create a multiprocessing pool with one worker per specified GPU
#         with mp.Pool(processes=num_gpus) as pool:
#             results = pool.starmap(self.process_on_gpu, [(chunk, device_id) for chunk, device_id in zip(pair_chunks, self.device_ids)])

#         # Flatten the list of results from all processes
#         return [score for result in results for score in result]


# # Usage example
# if __name__ == "__main__":
#     model_name = 'BAAI/bge-reranker-v2-gemma'
#     adapter_path = "/home/mabdallah/TrialMatchAI/src/Matcher/finetuned_reranker/"
    
#     # Initialize Reranker instance with specific device IDs and batch size
#     reranker = Reranker(model_name, adapter_path, device_ids=[0, 1], batch_size=5)

#     # Pairs of (query, passage) data
#     pairs = [
#         ['Patient with KRAS mutation', 'Patient without KRAS mutation'],
#         ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']
#     ] * 1000

#     # Compute the score
#     scores = reranker.compute_score(pairs)
#     print(scores)


