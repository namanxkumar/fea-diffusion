python calculate_accuracy_qualitative.py ^
--ground_truth_data_dir data/qualnoholes ^
--generated_samples_dir results/256/qualnoholes183 ^
--num_plates 100 ^
--num_conditions_per_plate 1 ^
--num_steps_per_condition 2 ^
--image_size 256

python calculate_accuracy_qualitative.py ^
--ground_truth_data_dir data/qualholes ^
--generated_samples_dir results/256/qualholes183 ^
--num_plates 100 ^
--num_conditions_per_plate 1 ^
--num_steps_per_condition 2 ^
--image_size 256

python calculate_accuracy_qualitative.py ^
--ground_truth_data_dir data/qualforceramp ^
--generated_samples_dir results/256/qualforceramp183 ^
--num_plates 100 ^
--num_conditions_per_plate 1 ^
--num_steps_per_condition 11 ^
--image_size 256

python calculate_accuracy_qualitative.py ^
--ground_truth_data_dir data/qualitativesample ^
--generated_samples_dir results/256/256ckpt183 ^
--num_plates 200 ^
--num_conditions_per_plate 1 ^
--num_steps_per_condition 2 ^
--image_size 256