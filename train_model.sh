#!/bin/bash
python train_model.py \
--data_dir data/feadata \
--sample_data_dir data/sample_data_1 \
--num_sample_conditions_per_plate 4 \
--results_dir results \
--image_size 64 \
--batch_size \
--num_gradient_accumulation_steps \
--num_steps 6 \
--num_steps_per_milestone \
--learning_rate \
--loss_type \
--use_wandb \
--wandb_project \
--wandb_restrict_cache 