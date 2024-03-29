#!/bin/bash
python generate_data.py \
--num_plates 2500 \
--data_dir data/feadata2500 \
--start_plate 1 \
--conditions_per_plate 4 \
--steps_per_condition 6 \
--image_size 512 \
--save_meshes \
--save_displacement  