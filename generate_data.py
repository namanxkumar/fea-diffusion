from datagen.generate import generate_data
import argparse

parser = argparse.ArgumentParser(description='Generate data for training.')
parser.add_argument('--num_plates', type=int, default=1, help='Number of plates to generate.')
parser.add_argument('--start_plate', type=int, default=None, help='Plate index to start generating from.')
parser.add_argument('--conditions_per_plate', type=int, default=4, help='Number of conditions to generate per plate.')
parser.add_argument('--steps_per_condition', type=int, default=11, help='Number of steps to generate per condition.')
parser.add_argument('--mesh_size', type=int, default=1e-2, help='Mesh size.')
parser.add_argument('--image_size', type=int, default=512, help='Image size.')
parser.add_argument('--save_displacement', action='store_true', help='Save displacement images.')
parser.add_argument('--save_strain', action='store_true', help='Save strain images.')
parser.add_argument('--save_stress', action='store_true', help='Save stress images.')
parser.add_argument('--data_dir', type=str, default='data', help='Data directory.')
args = parser.parse_args()

assert args.save_displacement or args.save_strain or args.save_stress, 'Must save at least one of displacement, strain, or stress.'

generate_data(
    data_dir=args.data_dir,
    image_size=args.image_size,
    num_plates=args.num_plates,
    start_plate=args.start_plate,
    conditions_per_plate=args.conditions_per_plate,
    mesh_size=args.mesh_size,
    save_displacement=args.save_displacement,
    save_strain=args.save_strain,
    save_stress=args.save_stress,
    num_steps_per_condition=args.steps_per_condition
)

# generate_data(
#     data_dir="data",
#     image_size=512,
#     num_plates=500,
#     start_plate=None, # None for starting from scratch, or a number to continue from
#     conditions_per_plate=4,
#     mesh_size=1e-2,
#     save_displacement=True,
#     save_strain=False,
#     save_stress=False
# )