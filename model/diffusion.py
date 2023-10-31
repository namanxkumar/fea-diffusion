from .unet import UNet

from pathlib import Path

import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np

from PIL import Image

from functools import partial
import math

from einops.layers.torch import Rearrange
from einops import repeat, rearrange

from typing import Optional, Dict, Tuple, Union, cast

# In regular diffusion implementations, a groundtruth image is provided from the dataset, a random timestep is generated for each forward pass,
# the corresponding amount of noise is added to the groundtruth image to degrade it, and the predicted image is sampled from the model.
# 
# In our case however, the dataset contains both the groundtruth (final) image to be generated as well as the intermediate images, so we can
# use the intermediate images as the groundtruth images for each timestep. This means that each intermediate image and corresponding timestep is
# provided as a separate datapoint sampled from the dataset.
# 
# In this case then, the diffusion class is not necessary and we can just create a Trainer class directly.

class FEADataset(Dataset):
    def __init__(
            self, 
            folder: str, 
            extension: str = 'png', 
            image_size = 256, 
            augmentation: bool = True, 
            conditions_per_plate: int = 4, 
            num_steps: int = 11, 
            # displacement: bool = True, 
            # strain: bool = False, 
            # stress: bool = False, 
            min_max_magnitude: Tuple[int, int] = (500, 5000)
        ):
        super().__init__()
        self.path = Path(f'{folder}')
        assert self.path.exists(), f'Error: Dataset directory {self.path} does not exist.'
        self.extension = extension
        self.image_size = image_size
        self.augmentation = augmentation

        self.number_of_plate_geometries = len(list(self.path.glob('*')))
        
        self.conditions_per_plate_geometry = conditions_per_plate
        
        self.num_steps = num_steps - 1 # 0th index step is not used

        self.samples_per_plate = self.conditions_per_plate_geometry * self.num_steps

        self.total_samples = self.number_of_plate_geometries * self.samples_per_plate

        # self.displacement = displacement
        # self.strain = strain
        # self.stress = stress
        self.min_max_magnitude = min_max_magnitude

    def normalize_by_division(self, tensor: Tensor, value: float) -> Tensor:
        return tensor / value
    
    def normalize_to_negative_one_to_one(self, tensor: Tensor) -> Tensor:
        return tensor * 2.0 - 1.0
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, index: int) -> Dict[str, Union[Tensor, Tuple[Tensor, Tensor]]]:
        plate_index = (index // (self.samples_per_plate)) + 1
        condition_index = (index % (self.samples_per_plate)) // self.num_steps + 1
        step_index = (index % (self.samples_per_plate)) % self.num_steps + 1

        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.Grayscale(),
            transforms.RandomHorizontalFlip() if self.augmentation else transforms.Lambda(lambda x: x),
            transforms.RandomVerticalFlip() if self.augmentation else transforms.Lambda(lambda x: x),
            transforms.PILToTensor(),
            transforms.Lambda(lambda x: self.normalize_by_division(x, 255.0)),
        ])
        
        sample = {}
        if step_index == 1:
            sample['previous_iteration'] = (
                self.normalize_to_negative_one_to_one(transform(Image.open(self.path / f'{plate_index}' / f'input.{self.extension}'))),
            )*2
        else:
            sample['previous_iteration'] = (
                self.normalize_to_negative_one_to_one(transform(Image.open(self.path / f'{plate_index}' / f'{condition_index}' / f'outputs_displacement_x_{step_index - 1}.{self.extension}'))), 
                self.normalize_to_negative_one_to_one(transform(Image.open(self.path / f'{plate_index}' / f'{condition_index}' / f'outputs_displacement_y_{step_index - 1}.{self.extension}')))
            )
        
        # if self.displacement:
        sample['displacement'] = (
            self.normalize_to_negative_one_to_one(transform(Image.open(self.path / f'{plate_index}' / f'{condition_index}' / f'outputs_displacement_x_{step_index}.{self.extension}'))), 
            self.normalize_to_negative_one_to_one(transform(Image.open(self.path / f'{plate_index}' / f'{condition_index}' / f'outputs_displacement_y_{step_index}.{self.extension}')))
        ) 
        # if self.strain:
        #     sample['strain'] = (transform(Image.open(self.path / f'{plate_index}' / f'{condition_index}' / f'outputs_strain_x_{step_index}.{self.extension}')), transform(Image.open(self.path / f'{plate_index}' / f'{condition_index}' / f'outputs_strain_y_{step_index}.{self.extension}')))
        # if self.stress:
        #     sample['stress'] = (transform(Image.open(self.path / f'{plate_index}' / f'{condition_index}' / f'outputs_stress_x_{step_index}.{self.extension}')), transform(Image.open(self.path / f'{plate_index}' / f'{condition_index}' / f'outputs_stress_y_{step_index}.{self.extension}')))
        
        sample['constraints'] = transform(Image.open(self.path / f'{plate_index}' / f'{condition_index}' / f'regions_constraints.{self.extension}'))

        with open(self.path / f'{plate_index}' / f'{condition_index}' / f'magnitudes.txt', 'r') as f:
            magnitudes = list(map(lambda x: tuple(x.strip().split(':')), f.readlines()))
        
        forces = []

        for name, values in magnitudes:
            force_tensor = transform(Image.open(self.path / f'{plate_index}' / f'{condition_index}' / f'regions_{name}.{self.extension}'))
            normalized_magnitude = tuple(map(lambda value: np.sign(value) * ((float(abs(value)) - self.min_max_magnitude[0]) / (self.min_max_magnitude[1] - self.min_max_magnitude[0])), values))
            forces.append((force_tensor * normalized_magnitude[0], force_tensor * normalized_magnitude[1]))

        # TODO: Combine all forces into one tensor
        sample['forces'] = torch.sum(torch.stack(forces, dim = 0), dim = 0) / len(forces)
        
        return sample

class Trainer():
    def __init__(
            self,
            model: nn.Module,
            dataset_folder: str,
            augmentation: bool = True,
            train_batch_size: int = 16,
            gradient_accumulation_steps: int = 1,
            train_learning_rate: float = 1e-4,
            train_num_steps: int = 11,
            adam_betas = (0.9, 0.99),
            num_samples: int = 8,
            results_folder: str = 'results',
    ):
        super().__init__()
        pass

    def sample_iteration(self, previous_iteration, t):
        pass

    def calculate_losses(self, sampled_iteration, groundtruth_iteration):
        pass

    def train(self):
        pass