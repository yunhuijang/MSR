python model/generalist_generator.py \
    --k 10 \
    --task mol2text \
    --wandb_mode online \
    --cot_mode_aromatic \
    --cot_mode_chain \
    --cot_mode_con_ring_name \
    --cot_mode_functional_group \
    --model_id meta-llama/Meta-Llama-3.1-70B-Instruct \
    --architecture llama3