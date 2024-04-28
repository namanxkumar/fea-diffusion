from metrics.calculate_accuracy_qualitative import calculate_accuracy, calculate_accuracy_per_step
import argparse
import numpy as np
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

MAE, MSE, RMSE, SSIM, mean_absolute_error_values, mean_squared_error_values, root_mean_squared_error_values, ssim = calculate_accuracy(
    num_plates=args.num_plates,
    num_conditions_per_plate=args.num_conditions_per_plate,
    num_steps=args.num_steps_per_condition,
    image_size=args.image_size,
    ground_truth_data_dir=args.ground_truth_data_dir,
    generated_samples_dir=args.generated_samples_dir,
)

def round_off(value):
    return np.around(value, decimals=4)

print("Mean absolute error: {}".format(round_off(MAE)))
print("Mean squared error: {}".format(round_off(MSE)))
print("RMSE: {}".format(round_off(RMSE)))
print("SSIM: {}".format(round_off(SSIM)))
print("Mean absolute error: {}".format(round_off(mean_absolute_error_values)))
print("Mean squared error: {}".format(round_off(mean_squared_error_values)))
print("Root mean squared error: {}".format(round_off(root_mean_squared_error_values)))
print("SSIM: {}".format(round_off(ssim)))