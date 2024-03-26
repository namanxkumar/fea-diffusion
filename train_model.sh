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
--num_steps_per_milestone 500 \
--loss_type "l2" \
--use_wandb \
--wandb_project "fea_diffusion_2"
# --wandb_restrict_cache 
# --learning_rate 