from .accuracy_function import calculate_accuracy_for_one_sample
from pathlib import Path
import numpy as np
from tqdm import tqdm


def calculate_accuracy(
    num_plates,
    num_conditions_per_plate,
    num_steps,
    image_size,
    ground_truth_data_dir,
    generated_samples_dir,
    progress_bar=True,
):
    assert num_steps > 1, "Must have at least 2 steps per condition."

    total_samples = num_plates * num_conditions_per_plate * (num_steps - 1)

    ground_truth_data_path = Path(ground_truth_data_dir)
    generated_samples_path = Path(generated_samples_dir)

    assert (
        ground_truth_data_path.exists()
    ), "Ground truth data directory does not exist."
    assert (
        generated_samples_path.exists()
    ), "Generated samples directory does not exist."

    mean_absolute_error_values = np.zeros(total_samples)
    mean_squared_error_values = np.zeros(total_samples)
    root_mean_squared_error_values = np.zeros(total_samples)

    if progress_bar:
        total_samples = tqdm(range(total_samples), desc="Calculating accuracy")
    else:
        total_samples = range(total_samples)

    for index in total_samples:
        plate_index = (index // (num_conditions_per_plate * (num_steps - 1))) + 1
        condition_index = (index % (num_conditions_per_plate * (num_steps - 1))) // (
            num_steps - 1
        ) + 1
        step_index = (index % (num_conditions_per_plate * (num_steps - 1))) % (
            num_steps - 1
        ) + 1

        if num_steps <= 10:
            domainfilename = "domain.{}.vtk"
        else:
            domainfilename = "domain.{:0>2}.vtk"

        mesh_path = (
            ground_truth_data_path
            / str(plate_index)
            / str(condition_index)
            / domainfilename.format(step_index)
        )
        x_displacement_path = (
            generated_samples_path
            / str(plate_index)
            / str(condition_index)
            / "sample_x_{}.png".format(step_index)
        )
        y_displacement_path = (
            generated_samples_path
            / str(plate_index)
            / str(condition_index)
            / "sample_y_{}.png".format(step_index)
        )
        geometry_path = ground_truth_data_path / str(plate_index) / "input.png"

        (
            mean_absolute_error_values[index],
            mean_squared_error_values[index],
            root_mean_squared_error_values[index],
        ) = calculate_accuracy_for_one_sample(
            mesh_path,
            x_displacement_path,
            y_displacement_path,
            geometry_path,
            image_size=image_size,
        )

    return (
        mean_absolute_error_values,
        mean_squared_error_values,
        root_mean_squared_error_values,
        np.mean(mean_absolute_error_values),
        np.mean(mean_squared_error_values),
        np.mean(root_mean_squared_error_values),
    )


def calculate_accuracy_per_step(
    num_plates,
    num_conditions_per_plate,
    num_steps,
    image_size,
    ground_truth_data_dir,
    generated_samples_dir,
    progress_bar=True,
):
    assert num_steps > 1, "Must have at least 2 steps per condition."

    total_samples = num_plates * num_conditions_per_plate * (num_steps - 1)

    ground_truth_data_path = Path(ground_truth_data_dir)
    generated_samples_path = Path(generated_samples_dir)

    assert (
        ground_truth_data_path.exists()
    ), "Ground truth data directory does not exist."
    assert (
        generated_samples_path.exists()
    ), "Generated samples directory does not exist."

    # mean_absolute_error_values = np.zeros(total_samples)
    # mean_squared_error_values = np.zeros(total_samples)
    # root_mean_squared_error_values = np.zeros(total_samples)
    mean_absolute_error_values_per_step = np.zeros((total_samples, num_steps - 1))
    mean_squared_error_values_per_step = np.zeros((total_samples, num_steps - 1))
    root_mean_squared_error_values_per_step = np.zeros((total_samples, num_steps - 1))

    if progress_bar:
        total_samples = tqdm(range(total_samples), desc="Calculating accuracy")
    else:
        total_samples = range(total_samples)

    for index in total_samples:
        plate_index = (index // (num_conditions_per_plate * (num_steps - 1))) + 1
        condition_index = (index % (num_conditions_per_plate * (num_steps - 1))) // (
            num_steps - 1
        ) + 1
        step_index = (index % (num_conditions_per_plate * (num_steps - 1))) % (
            num_steps - 1
        ) + 1

        if num_steps <= 10:
            domainfilename = "domain.{}.vtk"
        else:
            domainfilename = "domain.{:0>2}.vtk"

        mesh_path = (
            ground_truth_data_path
            / str(plate_index)
            / str(condition_index)
            / domainfilename.format(step_index)
        )
        x_displacement_path = (
            generated_samples_path
            / str(plate_index)
            / str(condition_index)
            / "sample_x_{}.png".format(step_index)
        )
        y_displacement_path = (
            generated_samples_path
            / str(plate_index)
            / str(condition_index)
            / "sample_y_{}.png".format(step_index)
        )
        geometry_path = ground_truth_data_path / str(plate_index) / "input.png"

        (
            mean_absolute_error_values_per_step[index, step_index - 1],
            mean_squared_error_values_per_step[index, step_index - 1],
            root_mean_squared_error_values_per_step[index, step_index - 1],
        ) = calculate_accuracy_for_one_sample(
            mesh_path,
            x_displacement_path,
            y_displacement_path,
            geometry_path,
            image_size=image_size,
        )

    mean_absolute_error_values = np.mean(mean_absolute_error_values_per_step, axis=0)
    mean_squared_error_values = np.mean(mean_squared_error_values_per_step, axis=0)
    root_mean_squared_error_values = np.mean(
        root_mean_squared_error_values_per_step, axis=0
    )

    return (
        mean_absolute_error_values,
        mean_squared_error_values,
        root_mean_squared_error_values,
        np.mean(mean_absolute_error_values),
        np.mean(mean_squared_error_values),
        np.mean(root_mean_squared_error_values),
    )
