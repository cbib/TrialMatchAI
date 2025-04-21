# Set common variables
export MODEL_NAME='./finetuned_model_roberta/'
export OUTPUT_BASE_DIR=./output_eval # Base directory for saving output files
export BATCH_SIZE=32
export SEED=42
export EVAL_ACCUMULATION_STEPS=50  # Adjust this value based on your GPU memory capacity

# List of entities to evaluate
entities=("NCBI-disease" "BC4CHEMD" "JNLPBA-cl" "JNLPBA-ct" "JNLPBA-dna" "JNLPBA-rna")

# Loop over each entity and run evaluation
for ENTITY in "${entities[@]}"; do
    echo "Evaluating on entity: $ENTITY"
    
    # Set output directory for the current entity
    export OUTPUT_DIR="${OUTPUT_BASE_DIR}/${ENTITY}"
    mkdir -p $OUTPUT_DIR

    # Run evaluation
    python evaluate.py \
        --model_name_or_path $MODEL_NAME \
        --data_dir NERdata/ \
        --labels NERdata/$ENTITY/labels.txt \
        --output_dir $OUTPUT_DIR \
        --eval_data_type $ENTITY \
        --eval_data_list $ENTITY \
        --max_seq_length 128 \
        --per_device_eval_batch_size $BATCH_SIZE \
        --eval_accumulation_steps $EVAL_ACCUMULATION_STEPS \
        --seed $SEED \
        --do_eval \
        --do_predict
    
    echo "Results stored in: $OUTPUT_DIR"
done
