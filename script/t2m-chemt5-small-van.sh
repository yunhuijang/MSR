python model/one_stage_generator.py \
--architecture multitask-text-and-chemistry-t5-small-standarddarddarddarddarddard \
--wandb_mode online \
--train_batch_size 8 \
--eval_batch_size 8 \
--epochs 250 \
--model_id GT4SD \
--max_length 820 \
--generation_mode \
--max_new_tokens 256 \
--check_val_every_n_epoch 20 \
--weight_decay 0 \
--learning_rate 6e-4 \
--warmup_ratio 0 \
--lr_scheduler_type linear




