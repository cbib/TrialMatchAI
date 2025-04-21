import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
# from local_gemma import LocalGemma2ForCausalLM

def get_model(model_args, training_args):
    
    # model = LocalGemma2ForCausalLM.from_pretrained(model_args.model_name_or_path, 
    #                                                preset="auto", 
    #                                                attn_implementation='eager',
    #                                                cache_dir=model_args.cache_dir,
    #                                                torch_dtype=torch.float16 if training_args.fp16 else torch.bfloat16,
    #                                                token=model_args.token,
    #                                                from_tf=bool(".ckpt" in model_args.model_name_or_path),
    #                                                trust_remote_code=True,
    #                                                use_flash_attention_2=True if model_args.use_flash_attn else False)
    quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                            bnb_4bit_use_double_quant=True,
                                            bnb_4bit_quant_type="nf4",
                                            bnb_4bit_compute_dtype="float16",)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.float16 if training_args.fp16 else torch.bfloat16,
        token=model_args.token,
        cache_dir=model_args.cache_dir,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        quantization_config=quantization_config,
    )
    
    model = model.to('cuda')
    
    if torch.cuda.device_count() > 1:  # If more than 1 GPU
        model.is_parallelizable = True
        model.model_parallel = True
    
    model.config.use_cache = False

    if model_args.from_peft is not None:
        model = PeftModel.from_pretrained(model, model_args.from_peft, is_trainable=True)
        model.print_trainable_parameters()
    else:
        if model_args.use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=model_args.lora_rank,
                target_modules=model_args.target_modules,
                lora_alpha=model_args.lora_alpha,
                lora_dropout=model_args.lora_dropout,
                modules_to_save=model_args.lora_extra_parameters
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

    print(model)
    return model
