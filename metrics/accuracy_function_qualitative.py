import numpy as np
import pyvista as pv
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from SSIM_PIL import compare_ssim

def calculate_accuracy_for_one_sample(
    displacement_x_file, displacement_y_file, gt_x_file, gt_y_file, image_size
):
    predicted_displacement_x = np.array(
        ImageOps.grayscale(
            Image.open(displacement_x_file).resize((image_size, image_size))
        )
    ).squeeze()/255.0

    predicted_displacement_y = np.array(
        ImageOps.grayscale(
            Image.open(displacement_y_file).resize((image_size, image_size))
        )
    ).squeeze()/255.0

    gt_displacement_x = np.array(
        ImageOps.grayscale(Image.open(gt_x_file).resize((image_size, image_size)))
    ).squeeze()/255.0

    gt_displacement_y = np.array(
        ImageOps.grayscale(Image.open(gt_y_file).resize((image_size, image_size)))
    ).squeeze()/255.0

    mean_absolute_error = np.mean(
        [
            np.mean(np.abs(predicted_displacement_x - gt_displacement_x)),
            np.mean(np.abs(predicted_displacement_y - gt_displacement_y)),
        ]
    )

    mean_squared_error = np.mean(
        [
            np.mean(np.square(predicted_displacement_x - gt_displacement_x)),
            np.mean(np.square(predicted_displacement_y - gt_displacement_y)),
        ]
    )

 
    ssimx = compare_ssim((ImageOps.grayscale(Image.open(displacement_x_file).resize((image_size,image_size)))), 
                            (ImageOps.grayscale(Image.open(gt_x_file).resize((image_size,image_size)))), GPU=False)
    
    ssimy = compare_ssim((ImageOps.grayscale(Image.open(displacement_y_file).resize((image_size,image_size)))), 
                        (ImageOps.grayscale(Image.open(gt_y_file).resize((image_size,image_size)))), GPU=False)
                        
    ssim = np.mean([ssimx, ssimy])
    
    root_mean_squared_error = np.sqrt(mean_squared_error)

    return mean_absolute_error, mean_squared_error, root_mean_squared_error, ssim
