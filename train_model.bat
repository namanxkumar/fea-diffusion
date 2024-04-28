python train_model.py ^
--data_dir data/4016to5000 ^
--sample_data_dir data/sample_data_1 ^
--num_steps_per_sample_condition 6 ^
--num_steps_per_condition 6 ^
--num_sample_conditions_per_plate 4 ^
--results_dir results ^
--image_size 64 ^
--batch_size 6 ^
--num_gradient_accumulation_steps 3 ^
--num_steps 10000 ^
--num_steps_per_milestone 5 ^
 
