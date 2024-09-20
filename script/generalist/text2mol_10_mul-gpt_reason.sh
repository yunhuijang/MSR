python model/generalist_generator.py \
    --k 10 \
    --task text2mol \
    --wandb_mode online \
    --cot_mode multiset_formula-chain-aromatic-con_ring_name-func_simple-chiral \
    --model_id gpt-4o \
    --architecture gpt4o \
    --is_reason