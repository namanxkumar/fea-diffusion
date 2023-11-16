from pathlib import Path

from tqdm.autonotebook import tqdm

from .fdnunet import FDNUNet

import matplotlib.pyplot as plt

import torch
from torch import nn, Tensor
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from zipfile import ZipFile

import numpy as np

from accelerate import Accelerator

from ema_pytorch import EMA

from PIL import Image

import logging
from datetime import datetime

from typing import Optional, Dict, Tuple, List, Union

# In regular diffusion implementations, a groundtruth image is provided from the dataset, a random timestep is generated for each forward pass,
# the corresponding amount of noise is added to the groundtruth image to degrade it, and the predicted image is sampled from the model.
# 
# In our case however, the dataset contains both the groundtruth (final) image to be generated as well as the intermediate images, so we can
# use the intermediate images as the groundtruth images for each timestep. This means that each intermediate image and corresponding timestep is
# provided as a separate datapoint sampled from the dataset.
# 
# In this case then, the diffusion class is not necessary and we can just create a Trainer class directly.

def exists(value):
    return value is not None

class FEADataset(Dataset):
    def __init__(
            self, 
            folder: str, 
            extension: str = 'png', 
            image_size = 256, 
            augmentation: bool = False, 
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
    
    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        plate_index = (index // (self.samples_per_plate)) + 1
        condition_index = (index % (self.samples_per_plate)) // self.num_steps + 1
        step_index = (index % (self.samples_per_plate)) % self.num_steps + 1

        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.Grayscale(),
            # transforms.RandomHorizontalFlip() if self.augmentation else transforms.Lambda(lambda x: x),
            # transforms.RandomVerticalFlip() if self.augmentation else transforms.Lambda(lambda x: x),
            transforms.PILToTensor(),
            transforms.Lambda(lambda x: self.normalize_by_division(x, 255.0)),
            transforms.Lambda(lambda x: TF.invert(x)),
        ])
        
        sample = {}

        sample['geometry'] = transform(Image.open(self.path / f'{plate_index}' / f'input.{self.extension}'))
        sample['geometry'] = self.normalize_to_negative_one_to_one(F.threshold(torch.clamp(255*sample['geometry'], min=0, max=1.0), 0.5, 0.0).int()).float()
        sample['plate_index'] = torch.tensor(plate_index)
        sample['condition_index'] = torch.tensor(condition_index)
        sample['iteration_index'] = torch.tensor(step_index)

        if step_index == 1:
            sample['previous_iteration'] = (
                sample['geometry'],
            )*2
        else:
            sample['previous_iteration'] = (
                self.normalize_to_negative_one_to_one(transform(Image.open(self.path / f'{plate_index}' / f'{condition_index}' / f'outputs_displacement_x_{step_index - 1}.{self.extension}'))), 
                self.normalize_to_negative_one_to_one(transform(Image.open(self.path / f'{plate_index}' / f'{condition_index}' / f'outputs_displacement_y_{step_index - 1}.{self.extension}')))
            )
        sample['previous_iteration'] = torch.cat(sample['previous_iteration'], dim = 0)

        # if self.displacement:
        sample['displacement'] = (
            self.normalize_to_negative_one_to_one(transform(Image.open(self.path / f'{plate_index}' / f'{condition_index}' / f'outputs_displacement_x_{step_index}.{self.extension}'))), 
            self.normalize_to_negative_one_to_one(transform(Image.open(self.path / f'{plate_index}' / f'{condition_index}' / f'outputs_displacement_y_{step_index}.{self.extension}')))
        )
        sample['displacement'] = torch.cat(sample['displacement'], dim = 0)

        # if self.strain:
        #     sample['strain'] = (transform(Image.open(self.path / f'{plate_index}' / f'{condition_index}' / f'outputs_strain_x_{step_index}.{self.extension}')), transform(Image.open(self.path / f'{plate_index}' / f'{condition_index}' / f'outputs_strain_y_{step_index}.{self.extension}')))
        # if self.stress:
        #     sample['stress'] = (transform(Image.open(self.path / f'{plate_index}' / f'{condition_index}' / f'outputs_stress_x_{step_index}.{self.extension}')), transform(Image.open(self.path / f'{plate_index}' / f'{condition_index}' / f'outputs_stress_y_{step_index}.{self.extension}')))
        
        constraints = []
        condition_path = self.path / f'{plate_index}' / f'{condition_index}'
        for path in condition_path.iterdir():
            if path.match("*Constraint*"):
                constraints.append(transform(Image.open(path)))
        sample['constraints'] = self.normalize_to_negative_one_to_one(F.threshold(torch.clamp(255*torch.sum(torch.stack(constraints, dim = 0), dim = 0), min=0, max=1.0), 0.5, 0.0).int()).float()

        with open(self.path / f'{plate_index}' / f'{condition_index}' / f'magnitudes.txt', 'r') as f:
            magnitudes = list(map(lambda x: tuple(x.strip().split(':')), f.readlines()))
        
        forces = []

        for name, values in magnitudes:
            values = eval(values)
            force_tensor = transform(Image.open(self.path / f'{plate_index}' / f'{condition_index}' / f'regions_{name}.{self.extension}'))
            force_tensor = torch.clamp(255*force_tensor, min=0, max=1.0)
            normalized_magnitude = tuple(map(lambda value: np.sign(value) * ((float(abs(value)) - self.min_max_magnitude[0]) / (self.min_max_magnitude[1] - self.min_max_magnitude[0])) * (step_index/self.num_steps), values))
            forces.append(torch.cat((force_tensor * normalized_magnitude[0], force_tensor * normalized_magnitude[1]), dim = 0))

        # Combine all forces into one tensor
        sample['forces'] = torch.clamp(torch.sum(torch.stack(forces, dim = 0), dim = 0), min=-1.0, max=1.0)
        
        return sample

class Step():
    def __init__(self, step: int, gradient_accumulation_steps: int, batch_size: int):
        self.step = step
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.batch_size = batch_size

    def load_state_dict(self, state_dict: dict):
        self.step = state_dict['step']
        self.gradient_accumulation_steps = state_dict['gradient_accumulation_steps']
        self.batch_size = state_dict['batch_size'] if 'batch_size' in state_dict else self.batch_size

    def state_dict(self):
        state_dict = {
            'step': self.step,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'batch_size': self.batch_size,
        }
        return state_dict

class Trainer():
    def __init__(
            self,
            model: nn.Module,
            dataset_folder: str,
            sample_dataset_folder: str,
            dataset_image_size: int = 256,
            use_dataset_augmentation: bool = False,
            train_batch_size: int = 16,
            sample_batch_size: Optional[int] = None,
            num_sample_conditions_per_plate: int = 1,
            num_gradient_accumulation_steps: int = 1,
            train_learning_rate: float = 1e-4,
            num_train_steps: int = 1000,
            num_steps_per_milestone: int = 100,
            adam_betas = (0.9, 0.99),
            max_gradient_norm: float = 1.0,
            ema_decay: float = 0.995,
            ema_steps_per_milestone: int = 10,
            results_folder: str = 'results',
            use_batch_split_over_devices: bool = True,
    ):
        super().__init__()
        self.accelerator = Accelerator(
            split_batches=use_batch_split_over_devices,
        )

        self.model = model
        
        # Parameters
        self.num_steps_per_milestone = num_steps_per_milestone
        self.sample_batch_size = sample_batch_size if exists(sample_batch_size) else train_batch_size
        
        self.train_batch_size = train_batch_size
        self.num_gradient_accumulation_steps = num_gradient_accumulation_steps

        self.max_gradient_norm = max_gradient_norm
        
        assert (train_batch_size * num_gradient_accumulation_steps) >= 16, f'your effective batch size (train_batch_size x num_gradient_accumulation_steps) should be at least 16 or above'

        self.num_train_steps = num_train_steps

        # Dataset
        self.image_size = dataset_image_size
        self.dataset = FEADataset(dataset_folder, image_size=dataset_image_size, augmentation=use_dataset_augmentation)
        self.sample_dataset = FEADataset(sample_dataset_folder, image_size=dataset_image_size, augmentation=False, conditions_per_plate=num_sample_conditions_per_plate)
        self.num_samples = len(self.sample_dataset)

        assert len(self.dataset) >= 100, 'you should have at least 100 samples in your folder. at least 10k images recommended'

        self.train_dataloader = DataLoader(self.dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=0, pin_memory=True)
        self.sample_dataloader = DataLoader(self.sample_dataset, batch_size=self.sample_batch_size, shuffle=False, num_workers=0, pin_memory=True)

        # Optimizer
        self.optimizer = Adam(self.model.parameters(), lr=train_learning_rate, betas=adam_betas)

        # Results
        if self.accelerator.is_main_process:
            self.ema = EMA(self.model, beta=ema_decay, update_every=ema_steps_per_milestone)
            self.ema.to(self.device)


        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        log_name = 'train-e{}-b{}-lr{}-{}.log'.format(num_train_steps, train_batch_size, str(train_learning_rate)[2:], datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        logging.basicConfig(filename=(self.results_folder / log_name), level=logging.INFO, format='%(asctime)s %(message)s', force=True)

        self.step = Step(0, num_gradient_accumulation_steps, train_batch_size)

        # Prepare mode, optimizer, dataloader with accelerator
        self.model, self.optimizer, self.train_dataloader, self.sample_dataloader = self.accelerator.prepare(self.model, self.optimizer, self.train_dataloader, self.sample_dataloader)

        self.accelerator.register_for_checkpointing(self.ema)
        self.accelerator.register_for_checkpointing(self.step)

        self.skipped_dataloader = None

        self.train_yielder = self.yield_data(self.train_dataloader)

    @property
    def device(self):
        return self.accelerator.device
    
    def old_save_checkpoint(self, milestone, old_step = True):
        if not self.accelerator.is_local_main_process:
            return
        
        checkpoint = {
            'step': self.step if old_step else self.step.state_dict,
            'model': self.accelerator.get_state_dict(self.model),
            'optimizer': self.optimizer.state_dict(),
            'ema': self.ema.state_dict(),
        }

        torch.save(checkpoint, str(self.results_folder / f'model-{milestone}.pt'))

    def save_checkpoint(self, milestone):
        self.accelerator.save_state(self.results_folder / f'model-{milestone}')

        with ZipFile(self.results_folder / f'model-{milestone}.zip', 'w') as zip:
            path = self.results_folder / f'model-{milestone}'
            for file in path.iterdir():
                zip.write(file, arcname=file.relative_to(self.results_folder))
        
        # Delete the folder
        for file in path.iterdir():
            file.unlink()
        path.rmdir()

    def old_load_checkpoint(self, milestone: int, old_step = True):
        accelerator = self.accelerator
        device = accelerator.device

        checkpoint = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model: nn.Module = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(checkpoint['model'])

        if old_step:
            self.step.step = checkpoint['step']
        else:
            self.step.load_state_dict(checkpoint['step'])
        self.step.gradient_accumulation_steps = self.num_gradient_accumulation_steps
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        if self.accelerator.is_main_process:
            self.ema.load_state_dict(checkpoint['ema'])

    def unzip_checkpoint(self, milestone: int):
        with ZipFile(self.results_folder / f'model-{milestone}.zip', 'r') as zip:
            zip.extractall(self.results_folder / f'model-{milestone}')

    def load_checkpoint(self, milestone: int, override_batch_size: Optional[int] = None):
        self.accelerator.load_state(self.results_folder / f'model-{milestone}')
        self.step.batch_size = override_batch_size if exists(override_batch_size) else self.step.batch_size
        num_skips = (self.step.step * self.step.gradient_accumulation_steps * self.step.batch_size) // self.train_batch_size
        self.skipped_dataloader = self.accelerator.skip_first_batches(self.train_dataloader, num_skips)
        self.train_yielder = self.yield_data(self.train_dataloader, self.skipped_dataloader)
        self.step.gradient_accumulation_steps = self.num_gradient_accumulation_steps
        self.step.batch_size = self.train_batch_size

    def calculate_losses(self, sampled_iteration: Tensor, groundtruth_iteration: Tensor) -> Tensor:
        return F.mse_loss(sampled_iteration, groundtruth_iteration)

    @staticmethod
    def yield_data(dataloader, skipped_dataloader = None) -> Dict[str, Tensor]:
        if exists(skipped_dataloader):
            for data in skipped_dataloader:
                yield data
        while True:
            for data in dataloader:
                yield data
    
    @staticmethod
    def unnormalize_from_negative_one_to_one(tensor: Tensor) -> Tensor:
        return (tensor + 1.0) / 2.0
    
    @staticmethod
    def normalize_to_negative_one_to_one(tensor: Tensor) -> Tensor:
        return (tensor * 2.0) - 1.0

    def create_view_friendly_image(self, image: Tensor) -> Image.Image:
        image = image[None, ...]
        image = self.unnormalize_from_negative_one_to_one(image)
        # image = TF.invert(image)
        image = image * 255.0
        # image = image.repeat(3, 1, 1)
        # image = TF.to_pil_image(image, mode='F')
        # image = image.convert("RGB")
        return image

    def sample_model(self, sample: Dict[str, Tensor], use_ema_model: bool = False) -> Tensor:
        if type(self.model) == FDNUNet:
            conditions = torch.cat((sample['forces'], sample['constraints']), dim = 1).to(self.device)
        else:
            conditions = torch.cat((sample['forces'], sample['constraints'], sample['geometry']), dim = 1).to(self.device)
        previous_iteration = sample['previous_iteration'].to(self.device)
        iteration_index = sample['iteration_index'].to(self.device)
        if type(self.model) == FDNUNet:
            prediction = self.model(previous_iteration, iteration_index, conditions, sample['geometry'])
        else:
            prediction = self.model(previous_iteration, iteration_index, conditions) if not use_ema_model else self.ema.ema_model(previous_iteration, iteration_index, conditions)

        prediction = self.normalize_to_negative_one_to_one(self.unnormalize_from_negative_one_to_one(prediction) * self.unnormalize_from_negative_one_to_one(sample['geometry'])) # Mask out the regions that are not part of the geometry
        prediction = prediction * (1.0 - self.unnormalize_from_negative_one_to_one(sample['constraints'])) # Mask out the regions that are constrained
        return prediction

    def sample(self, batch, use_ema_model: bool = False) -> Tuple[List[Image.Image], Tensor]:
        with torch.inference_mode():
            output = self.sample_model(batch, use_ema_model)
            loss = self.calculate_losses(output, batch['displacement'])
            images = []
            for i in range(output.shape[0]):
                for j in range(output.shape[1]):
                    images.append(self.create_view_friendly_image(output[i][j]))
            return images, loss
    
    def sample_and_save(self, milestone: Union[int, str] = None, use_ema_model: bool = False):
        sampled_images: List[Image.Image] = []
        total_sample_loss = 0.0
        num_batches = 0
        for batch in tqdm(self.sample_dataloader):
            images, loss = self.sample(batch, use_ema_model)
            sampled_images += images
            total_sample_loss += loss
            num_batches += 1
        total_sample_loss /= num_batches

        for i, image in enumerate(sampled_images):
            if exists(milestone):
                plt.imsave(str(self.results_folder / f'sample-{i}-{milestone}.png'), torch.squeeze(image).cpu().detach().numpy(), cmap='Greys', vmin=0, vmax=255)
                # image.save(str(self.results_folder / f'sample-{i}-{milestone}.png'))
            else:
                plt.imsave(str(self.results_folder / f'sample-{i}.png'), torch.squeeze(image).cpu().detach().numpy(), cmap='Greys', vmin=0, vmax=255)
                # image.save(str(self.results_folder / f'sample-{i}.png'))

        return sampled_images, total_sample_loss
        
    def train(self, wandb_inject_function = None):
        print('Epoch Size: {} effective batches'.format((len(self.train_dataloader)/(self.num_gradient_accumulation_steps))))
        print('Number of Effective Epochs: {}'.format(self.num_train_steps/(len(self.train_dataloader)/(self.num_gradient_accumulation_steps))))
        with tqdm(initial = self.step.step, total = self.num_train_steps, disable = not self.accelerator.is_main_process) as progress_bar:
            while self.step.step < self.num_train_steps:
                total_loss = 0.0
                # with tqdm(initial = 0, total = self.num_gradient_accumulation_steps, disable = not self.accelerator.is_main_process, leave=False) as inner_progress_bar:
                for _ in range(self.num_gradient_accumulation_steps):
                    sample: dict = next(self.train_yielder)
                    output = self.sample_model(sample)

                    loss = self.calculate_losses(output, sample['displacement'])
                    loss = loss / self.num_gradient_accumulation_steps
                    total_loss += loss.item()
                    # inner_progress_bar.set_description(f'current batch loss: {total_loss:.4f}')
                    self.accelerator.backward(loss)
                    # inner_progress_bar.update(1)

                progress_bar.set_description(f'loss: {total_loss:.4f} ')
                
                logging.info(f'step: {self.step.step}, loss: {total_loss:.4f}')

                self.accelerator.wait_for_everyone()
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_gradient_norm) # Gradient clipping

                self.optimizer.step()
                self.optimizer.zero_grad()

                self.accelerator.wait_for_everyone()

                self.step.step += 1

                if self.accelerator.is_main_process:
                    
                    self.ema.update()
                    total_sample_loss = None
                    sampled_images = None
                    milestone = None

                    if self.step.step != 0 and self.step.step % self.num_steps_per_milestone == 0:
                        self.ema.ema_model.eval()
                        
                        with torch.inference_mode():
                            milestone = self.step.step // self.num_steps_per_milestone
                            sampled_images, total_sample_loss = self.sample_and_save(use_ema_model=True)
                            logging.info(f'sample loss: {total_sample_loss:.4f}')
                        self.save_checkpoint(milestone)

                    if exists(wandb_inject_function):
                        wandb_inject_function(self.step.step, total_loss, total_sample_loss, sampled_images, milestone)
                        
                progress_bar.update(1)
                
        self.accelerator.wait_for_everyone()
        self.save_checkpoint('final')
        self.accelerator.print('Training done!')