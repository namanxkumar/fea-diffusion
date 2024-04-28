python calculate_accuracy_qualitative.py ^
--ground_truth_data_dir data/testdata ^
--generated_samples_dir results/donkey256prev ^
--num_plates 200 ^
--num_conditions_per_plate 1 ^
--num_steps_per_condition 2 ^
--image_size 256

@REM python calculate_accuracy_qualitative.py ^
@REM --ground_truth_data_dir data/qualholes ^
@REM --generated_samples_dir results/256/qualholes183 ^
@REM --num_plates 100 ^
@REM --num_conditions_per_plate 1 ^
@REM --num_steps_per_condition 2 ^
@REM --image_size 256

@REM python calculate_accuracy_qualitative.py ^
@REM --ground_truth_data_dir data/qualforceramp ^
@REM --generated_samples_dir results/256/qualforceramp183 ^
@REM --num_plates 100 ^
@REM --num_conditions_per_plate 1 ^
@REM --num_steps_per_condition 11 ^
@REM --image_size 256

@REM python calculate_accuracy_qualitative.py ^
@REM --ground_truth_data_dir data/qualitativesample ^
@REM --generated_samples_dir results/256/256ckpt183 ^
@REM --num_plates 200 ^
@REM --num_conditions_per_plate 1 ^
@REM --num_steps_per_condition 2 ^
@REM --image_size 256