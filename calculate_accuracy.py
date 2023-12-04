from metrics.calculate_accuracy import calculate_accuracy
import argparse

parser = argparse.ArgumentParser(description="Calculate accuracy.")
parser.add_argument(
    "--ground_truth_data_dir", type=str, default="data/sample_data", help="Data directory."
)
parser.add_argument(
    "--generated_samples_dir", type=str, default="results/generated_data", help="Sample data directory."
)
parser.add_argument(
    "--num_plates", type=int, default=1, help="Number of plates in sample dataset."
)
parser.add_argument(
    "--num_conditions_per_plate",
    type=int,
    default=1,
    help="Number of conditions per plate in sample dataset.",
)
parser.add_argument(
    "--num_steps_per_condition",
    type=int,
    default=3,
    help="Number of steps per condition in sample dataset.",
)
parser.add_argument("--image_size", type=int, default=256, help="Image size.")

args = parser.parse_args()

_, _, _, mean_absolute_error_values, mean_squared_error_values, root_mean_squared_error_values = calculate_accuracy(
    num_plates=args.num_plates,
    num_conditions_per_plate=args.num_conditions_per_plate,
    num_steps=args.num_steps_per_condition,
    image_size=args.image_size,
    ground_truth_data_dir=args.ground_truth_data_dir,
    generated_samples_dir=args.generated_samples_dir,
)

print("Mean absolute error: {}".format(mean_absolute_error_values))
print("Mean squared error: {}".format(mean_squared_error_values))
print("Root mean squared error: {}".format(root_mean_squared_error_values))