import logging
from datetime import datetime

# from linecache import getline
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from accelerate import Accelerator

# from ema_pytorch import EMA
from PIL import Image
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm.autonotebook import tqdm

# from .fdnunet import FDNUNet
from .fdnunetwithaux import FDNUNetWithAux

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
        extension: str = "png",
        image_size=256,
        augmentation: bool = False,
        start_plate_index: int = 1,
        conditions_per_plate: int = 4,
        num_steps: int = 11,
        min_max_magnitude: Optional[
            Tuple[int, int]
        ] = None,  # If not provided, log scaling is used
        min_max_youngs_modulus: Optional[
            Tuple[int, int]
        ] = None,  # If not provided, log scaling is used
    ):
        super().__init__()
        self.path = Path(f"{folder}")
        assert (
            self.path.exists()
        ), f"Error: Dataset directory {self.path} does not exist."
        assert num_steps >= 2, "num_steps must be >= 2"
        self.extension = extension
        self.image_size = image_size
        self.augmentation = augmentation

        self.start_plate_index = start_plate_index

        self.number_of_plate_geometries = len(
            [directory for directory in self.path.iterdir() if directory.is_dir()]
        )

        self.conditions_per_plate_geometry = conditions_per_plate

        self.num_steps = num_steps - 1  # 0th index step is not used

        self.samples_per_plate = self.conditions_per_plate_geometry * self.num_steps

        self.total_samples = self.number_of_plate_geometries * self.samples_per_plate

        self.min_max_magnitude = min_max_magnitude
        self.min_max_youngs_modulus = min_max_youngs_modulus

    def normalize_by_division(self, tensor: Tensor, value: float) -> Tensor:
        return tensor / value

    def normalize_to_negative_one_to_one(self, tensor: Tensor) -> Tensor:
        return tensor * 2.0 - 1.0

    def unnormalize_from_negative_one_to_one(self, tensor: Tensor) -> Tensor:
        return (tensor + 1.0) / 2.0

    @staticmethod
    def _scale_min_max(value: float, min_max: Tuple[float, float]) -> float:
        return (value - min_max[0]) / (min_max[1] - min_max[0])

    @staticmethod
    def _scale_log(value: float) -> float:
        return np.log(value + 1).item()

    def __len__(self):
        return self.total_samples

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        # GET PLATE INDEX, CONDITION INDEX, AND STEP INDEX

        plate_index = (index // (self.samples_per_plate)) + self.start_plate_index
        condition_index = (index % (self.samples_per_plate)) // self.num_steps + 1
        step_index = (index % (self.samples_per_plate)) % self.num_steps + 1
        sample = {}
        sample["plate_index"] = torch.tensor(plate_index)
        sample["condition_index"] = torch.tensor(condition_index)
        sample["iteration_index"] = torch.tensor(step_index)

        # DEFINE TRANSFORMS

        transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.Grayscale(),
                # transforms.RandomHorizontalFlip() if self.augmentation else transforms.Lambda(lambda x: x),
                # transforms.RandomVerticalFlip() if self.augmentation else transforms.Lambda(lambda x: x),
                transforms.PILToTensor(),
                transforms.Lambda(lambda x: self.normalize_by_division(x, 255.0)),
                transforms.Lambda(lambda x: TF.invert(x)),
            ]
        )

        # LOAD IMAGES

        # GEOMETRY IMAGE

        sample["geometry"] = transform(
            Image.open(self.path / f"{plate_index}" / f"input.{self.extension}")
        )
        sample["geometry"] = self.normalize_to_negative_one_to_one(
            F.threshold(
                torch.clamp(255 * sample["geometry"], min=0, max=1.0), 0.5, 0.0
            ).int()
        ).float()

        # PREVIOUS ITERATION IMAGE

        # if step_index == 1:
        # sample["previous_iteration"] = (sample["geometry"],) * 2
        # else:
        # sample["previous_iteration"] = (
        #     self.normalize_to_negative_one_to_one(
        #         transform(
        #             Image.open(
        #                 self.path
        #                 / f"{plate_index}"
        #                 / f"{condition_index}"
        #                 / f"outputs_displacement_x_0.{self.extension}"
        #             )
        #         )
        #     ),
        #     self.normalize_to_negative_one_to_one(
        #         transform(
        #             Image.open(
        #                 self.path
        #                 / f"{plate_index}"
        #                 / f"{condition_index}"
        #                 / f"outputs_displacement_y_0.{self.extension}"
        #             )
        #         )
        #     ),
        # )
        # sample["previous_iteration"] = torch.cat(sample["previous_iteration"], dim=0)

        # DISPLACEMENT IMAGE

        sample["displacement"] = (
            self.normalize_to_negative_one_to_one(
                transform(
                    Image.open(
                        self.path
                        / f"{plate_index}"
                        / f"{condition_index}"
                        / f"outputs_displacement_x.{self.extension}"
                        # / f"outputs_displacement_x_{step_index}.{self.extension}"
                    )
                )
            ),
            self.normalize_to_negative_one_to_one(
                transform(
                    Image.open(
                        self.path
                        / f"{plate_index}"
                        / f"{condition_index}"
                        / f"outputs_displacement_y.{self.extension}"
                        # / f"outputs_displacement_y_{step_index}.{self.extension}"
                    )
                )
            ),
        )
        sample["displacement"] = torch.cat(sample["displacement"], dim=0)

        # CONSTRAINTS IMAGE

        constraints = []
        condition_path = self.path / f"{plate_index}" / f"{condition_index}"
        for path in condition_path.iterdir():
            if path.match("*Constraint*"):
                constraints.append(transform(Image.open(path)))
        sample["constraints"] = self.normalize_to_negative_one_to_one(
            F.threshold(
                torch.clamp(
                    255 * torch.sum(torch.stack(constraints, dim=0), dim=0),
                    min=0,
                    max=1.0,
                ),
                0.5,
                0.0,
            ).int()
        ).float()

        # FORCE IMAGES

        with open(
            self.path / f"{plate_index}" / f"{condition_index}" / "magnitudes.txt", "r"
        ) as f:
            magnitudes = list(map(lambda x: tuple(x.strip().split(":")), f.readlines()))

        edge_forces = []
        vertex_forces = []

        for name, values in magnitudes:
            name: str
            values: Tuple[float, float] = eval(values)

            force_tensor = transform(
                Image.open(
                    self.path
                    / f"{plate_index}"
                    / f"{condition_index}"
                    / f"regions_{name}.{self.extension}"
                )
            )

            # Ensure all values are either 0 or 1
            force_tensor = torch.clamp(255 * force_tensor, min=0, max=1.0)

            # Normalize magnitudes
            if exists(self.min_max_magnitude):
                normalized_magnitude = tuple(
                    map(
                        lambda value: self._scale_min_max(
                            float(abs(value)), self.min_max_magnitude
                        ),
                        values,
                    )
                )
            else:
                normalized_magnitude = tuple(
                    map(
                        lambda value: np.sign(value)
                        * (
                            self._scale_log(
                                float(abs(value) * ((step_index - 1) / self.num_steps))
                            )
                        ),
                        values,
                    )
                )

            # Multiply force tensor by normalized magnitude (1s are multiplied by the magnitude, 0s are left as 0s)
            if "Edge" in name:
                edge_forces.append(
                    torch.cat(
                        (
                            force_tensor * normalized_magnitude[0],
                            force_tensor * normalized_magnitude[1],
                        ),
                        dim=0,
                    )
                )
            elif "Vertex" in name:
                vertex_forces.append(
                    torch.cat(
                        (
                            force_tensor * normalized_magnitude[0],
                            force_tensor * normalized_magnitude[1],
                        ),
                        dim=0,
                    )
                )

        # Combine all forces into one tensor
        force_tensor = torch.zeros(sample["geometry"].shape)
        for force in edge_forces + vertex_forces:
            force_tensor = torch.where(force != 0, force, force_tensor)

        sample["forces"] = force_tensor

        # MATERIAL IMAGES

        with open(
            self.path / f"{plate_index}" / f"{condition_index}" / "materials.txt", "r"
        ) as f:
            regions = list(map(lambda x: tuple(x.strip().split(":")), f.readlines()))

        materials = []

        for name, values in regions:
            name: str
            youngs_modulus, poissons_ratio = eval(values)

            region_tensor = transform(
                Image.open(
                    self.path
                    / f"{plate_index}"
                    / f"{condition_index}"
                    / f"regions_{name}.{self.extension}"
                )
            )

            # Ensure all values are either 0 or 1
            region_tensor = torch.clamp(255 * region_tensor, min=0, max=1.0)

            # Check if region_tensor is all 0s
            if torch.sum(region_tensor) == 0:
                region_tensor = self.unnormalize_from_negative_one_to_one(
                    sample["geometry"]
                ).float()

            # Normalize youngs modulus and poisson's ratio
            normalized_youngs_modulus = (
                (
                    np.sign(youngs_modulus)
                    * self._scale_min_max(
                        float(abs(youngs_modulus)), self.min_max_youngs_modulus
                    )
                )
                if exists(self.min_max_youngs_modulus)
                else (
                    np.sign(youngs_modulus)
                    * self._scale_log(float(abs(youngs_modulus)))
                )
            )
            normalized_poissons_ratio = float(poissons_ratio)

            materials.append(
                torch.cat(
                    (
                        region_tensor * float(normalized_youngs_modulus),
                        region_tensor * normalized_poissons_ratio,
                    ),
                    dim=0,
                )
            )
        material_tensor = torch.zeros(sample["geometry"].shape)
        for material in materials:
            material_tensor = torch.where(material != 0, material, material_tensor)
        # sample["materials"] = torch.sum(torch.stack(materials, dim=0), dim=0)
        sample["materials"] = material_tensor

        path = str(self.path / f"{plate_index}" / f"{condition_index}" / "ranges.txt")
        with open(
            self.path / f"{plate_index}" / f"{condition_index}" / "ranges.txt", "r"
        ) as f:
            all_ranges = list(map(lambda x: tuple(x.strip().split(":")), f.readlines()))

        line = (step_index - 1) * 2

        ranges = []

        for index in [line, line + 1]:
            ranges += list(eval(all_ranges[index][1]))

        sample["displacement_range"] = torch.tensor(ranges, dtype=torch.float32)
        sample["log_displacement_range"] = torch.log(
            1 + torch.abs(sample["displacement_range"])
        )
        sample["sign_displacement_range"] = (
            (sample["displacement_range"] >= 0).int().float()
        )

        return sample


class Step:
    def __init__(self, step: int, gradient_accumulation_steps: int, batch_size: int):
        self.step = step
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.batch_size = batch_size

    def load_state_dict(self, state_dict: dict):
        self.step = state_dict["step"]
        self.gradient_accumulation_steps = state_dict["gradient_accumulation_steps"]
        self.batch_size = (
            state_dict["batch_size"] if "batch_size" in state_dict else self.batch_size
        )

    def state_dict(self):
        state_dict = {
            "step": self.step,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "batch_size": self.batch_size,
        }
        return state_dict


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        dataset_folder: str,
        sample_dataset_folder: str,
        dataset_image_size: int = 256,
        use_dataset_augmentation: bool = False,
        train_batch_size: int = 16,
        sample_batch_size: Optional[int] = None,
        train_start_plate_index: int = 1,
        sample_start_plate_index: int = 1,
        num_sample_conditions_per_plate: int = 1,
        num_steps_per_condition: int = 6,
        num_steps_per_sample_condition: int = 6,
        num_gradient_accumulation_steps: int = 1,
        train_learning_rate: float = 1e-4,
        num_train_steps: int = 1000,
        num_steps_per_milestone: int = 250,
        num_steps_per_soft_milestone: int = 50,
        adam_betas=(0.9, 0.99),
        max_gradient_norm: float = 1.0,
        loss_type: str = "l1",
        # ema_decay: float = 0.995,
        # ema_steps_per_milestone: int = 10,
        results_folder: str = "results",
        use_batch_split_over_devices: bool = True,
    ):
        super().__init__()
        assert num_steps_per_condition >= 2, "num_steps_per_condition must be >= 2"
        assert (
            num_steps_per_sample_condition >= 2
        ), "num_steps_per_sample_condition must be >= 2"

        self.accelerator = Accelerator(
            split_batches=use_batch_split_over_devices,
        )

        self.model = model

        # Parameters
        self.num_steps_per_milestone = num_steps_per_milestone
        self.num_steps_per_soft_milestone = num_steps_per_soft_milestone
        self.sample_batch_size = (
            sample_batch_size if exists(sample_batch_size) else train_batch_size
        )

        self.train_batch_size = train_batch_size
        self.num_gradient_accumulation_steps = num_gradient_accumulation_steps

        self.max_gradient_norm = max_gradient_norm

        assert (
            (train_batch_size * num_gradient_accumulation_steps) >= 16
        ), "your effective batch size (train_batch_size x num_gradient_accumulation_steps) should be at least 16 or above"

        self.num_train_steps = num_train_steps

        self.loss_type = loss_type

        # Dataset
        self.image_size = dataset_image_size
        self.dataset = FEADataset(
            dataset_folder,
            image_size=dataset_image_size,
            augmentation=use_dataset_augmentation,
            num_steps=num_steps_per_condition,
            start_plate_index=train_start_plate_index,
        )
        self.sample_dataset = FEADataset(
            sample_dataset_folder,
            image_size=dataset_image_size,
            augmentation=False,
            conditions_per_plate=num_sample_conditions_per_plate,
            num_steps=num_steps_per_sample_condition,
            start_plate_index=sample_start_plate_index,
        )
        self.num_samples = len(self.sample_dataset)

        assert (
            len(self.dataset) >= 100
        ), "you should have at least 100 samples in your folder. at least 10k images recommended"

        self.train_dataloader = DataLoader(
            self.dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        self.sample_dataloader = DataLoader(
            self.sample_dataset,
            batch_size=self.sample_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        # Optimizer
        self.optimizer = Adam(
            self.model.parameters(), lr=train_learning_rate, betas=adam_betas
        )

        # Results
        # if self.accelerator.is_main_process:
        #     self.ema = EMA(
        #         self.model, beta=ema_decay, update_every=ema_steps_per_milestone
        #     )
        #     self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        log_name = "train-e{}-b{}-lr{}-{}.log".format(
            num_train_steps,
            train_batch_size,
            str(train_learning_rate)[2:],
            datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
        )
        logging.basicConfig(
            filename=(self.results_folder / log_name),
            level=logging.INFO,
            format="%(asctime)s %(message)s",
            force=True,
        )

        self.step = Step(0, num_gradient_accumulation_steps, train_batch_size)

        # Prepare mode, optimizer, dataloader with accelerator
        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.sample_dataloader,
        ) = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.sample_dataloader
        )

        # self.accelerator.register_for_checkpointing(self.ema)
        self.accelerator.register_for_checkpointing(self.step)

        self.skipped_dataloader = None

        self.train_yielder = self.yield_data(self.train_dataloader)

    @property
    def device(self):
        return self.accelerator.device

    def old_save_checkpoint(self, milestone, old_step=True):
        if not self.accelerator.is_local_main_process:
            return

        checkpoint = {
            "step": self.step if old_step else self.step.state_dict,
            "model": self.accelerator.get_state_dict(self.model),
            "optimizer": self.optimizer.state_dict(),
            # "ema": self.ema.state_dict(),
        }

        torch.save(checkpoint, str(self.results_folder / f"model-{milestone}.pt"))

    def save_checkpoint(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        if milestone == "latest":
            self.delete_previous_and_rename_current_checkpoint_if_exists(milestone)
        else:
            self.delete_checkpoint_if_exists(milestone)

        self.accelerator.save_state(self.results_folder / f"model-{milestone}")

        with ZipFile(self.results_folder / f"model-{milestone}.zip", "w") as zip:
            path = self.results_folder / f"model-{milestone}"
            for file in path.iterdir():
                zip.write(file, arcname=file.relative_to(self.results_folder))

        # Delete the folder
        for file in path.iterdir():
            file.unlink()
        path.rmdir()

    def delete_checkpoint_if_exists(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        path = self.results_folder / f"model-{milestone}"
        if path.exists():
            for file in path.iterdir():
                file.unlink()
            path.rmdir()

        path = self.results_folder / f"model-{milestone}.zip"
        if path.exists():
            path.unlink()

    def delete_previous_and_rename_current_checkpoint_if_exists(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        path = self.results_folder / f"model-{milestone}-prev"
        if path.exists():
            for file in path.iterdir():
                file.unlink()
            path.rmdir()

        path = self.results_folder / f"model-{milestone}-prev.zip"
        if path.exists():
            path.unlink()

        # Rename
        path = self.results_folder / f"model-{milestone}"
        if path.exists():
            path.rename(self.results_folder / f"model-{milestone}-prev")

        path = self.results_folder / f"model-{milestone}.zip"
        if path.exists():
            path.rename(self.results_folder / f"model-{milestone}-prev.zip")

    def old_load_checkpoint(self, milestone: int, old_step=True):
        accelerator = self.accelerator
        device = accelerator.device

        checkpoint = torch.load(
            str(self.results_folder / f"model-{milestone}.pt"), map_location=device
        )

        model: nn.Module = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(checkpoint["model"])

        if old_step:
            self.step.step = checkpoint["step"]
        else:
            self.step.load_state_dict(checkpoint["step"])
        self.step.gradient_accumulation_steps = self.num_gradient_accumulation_steps
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        # if self.accelerator.is_main_process:
        #     self.ema.load_state_dict(checkpoint["ema"])

    def unzip_checkpoint(self, milestone: int):
        with ZipFile(self.results_folder / f"model-{milestone}.zip", "r") as zip:
            zip.extractall(self.results_folder / f"model-{milestone}")

    def load_checkpoint(
        self, milestone: int, override_batch_size: Optional[int] = None
    ):
        self.accelerator.load_state(self.results_folder / f"model-{milestone}")

        self.step.batch_size = (
            override_batch_size if exists(override_batch_size) else self.step.batch_size
        )

        num_skips = (
            self.step.step
            * self.step.gradient_accumulation_steps
            * self.step.batch_size
        ) // self.train_batch_size
        num_skips = num_skips % len(self.train_dataloader)

        self.skipped_dataloader = self.accelerator.skip_first_batches(
            self.train_dataloader, num_skips
        )

        self.train_yielder = self.yield_data(
            self.train_dataloader, self.skipped_dataloader
        )

        self.step.gradient_accumulation_steps = self.num_gradient_accumulation_steps
        self.step.batch_size = self.train_batch_size

    def calculate_losses(
        self,
        sampled_tensors: List[Tensor],
        groundtruth_tensors: List[Tensor],
    ) -> Tensor:
        # return F.l1_loss(sampled_iteration, groundtruth_iteration, reduction='sum')
        if self.loss_type == "l1":
            return torch.sum(
                torch.stack(
                    [
                        F.l1_loss(sampled_tensor, groundtruth_tensor)
                        for sampled_tensor, groundtruth_tensor in zip(
                            sampled_tensors, groundtruth_tensors
                        )
                    ]
                )
            )
        elif self.loss_type == "l2":
            return torch.sum(
                torch.stack(
                    [
                        F.mse_loss(sampled_tensor, groundtruth_tensor)
                        for sampled_tensor, groundtruth_tensor in zip(
                            sampled_tensors, groundtruth_tensors
                        )
                    ]
                )
            )
        else:
            raise NotImplementedError("Only l1 and l2 loss are supported")

    @staticmethod
    def yield_data(
        dataloader: DataLoader, skipped_dataloader: Optional[DataLoader] = None
    ):
        if exists(skipped_dataloader):
            for data in skipped_dataloader:
                data: Dict[str, Tensor]
                yield data
        while True:
            for data in dataloader:
                data: Dict[str, Tensor]
                yield data

    @staticmethod
    def unnormalize_from_negative_one_to_one(tensor: Tensor) -> Tensor:
        return (tensor + 1.0) / 2.0

    @staticmethod
    def normalize_to_negative_one_to_one(tensor: Tensor) -> Tensor:
        return (tensor * 2.0) - 1.0

    def create_view_friendly_image(self, image: Tensor) -> Tensor:
        image = image[None, ...]
        image = self.unnormalize_from_negative_one_to_one(image)
        image = image * 255.0
        return image

    def reverse_view_friendly_image(self, image: Tensor) -> Tensor:
        image = image[None, ...]
        image = image / 255.0
        image = self.normalize_to_negative_one_to_one(image)
        return image

    def sample_model(
        self,
        sample: Dict[str, Tensor],
        # use_ema_model: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        conditions = torch.cat((sample["forces"], sample["constraints"]), dim=1).to(
            self.device
        )
        primary_input = sample["materials"].to(self.device)

        if type(self.model) == FDNUNetWithAux:
            image_prediction, range_prediction = self.model(primary_input, conditions)
        else:
            image_prediction = self.model(primary_input, conditions)
            range_prediction = None
        # sample['geometry'] = sample['geometry'].to(prediction.device)
        # sample['constraints'] = sample['constraints'].to(prediction.device)
        image_prediction = self.normalize_to_negative_one_to_one(
            self.unnormalize_from_negative_one_to_one(image_prediction)
            * self.unnormalize_from_negative_one_to_one(sample["geometry"]),
        )  # Mask out the regions that are not part of the geometry
        # prediction = prediction * (1.0 - self.unnormalize_from_negative_one_to_one(sample['constraints'])) # Mask out the regions that are constrained
        return image_prediction, range_prediction

    def create_view_friendly_range(
        self, sign_range_output: Tensor, log_range_output: Tensor
    ) -> Tensor:
        sign = (sign_range_output < 0.5).int() * 2 - 1
        return sign * (torch.exp(log_range_output) - 1)

    def sample(self, batch) -> Tuple[List[Tensor], Optional[List[Tensor]], Tensor]:
        with torch.inference_mode():
            image_output, range_output = self.sample_model(batch)

            loss = self.calculate_losses(
                (
                    [image_output, *range_output]
                    if exists(range_output)
                    else [image_output]
                ),
                (
                    [
                        batch["displacement"],
                        batch["sign_displacement_range"],
                        batch["log_displacement_range"],
                    ]
                    if exists(range_output)
                    else [batch["displacement"]]
                ),
            )
            images = []
            ranges = []
            for batch_index in range(image_output.shape[0]):
                if exists(range_output):
                    ranges.append(
                        self.create_view_friendly_range(
                            *(
                                range[batch_index].clone().detach()
                                for range in range_output
                            )
                        )
                    )

                for channel_index in range(image_output.shape[1]):
                    images.append(
                        self.create_view_friendly_image(
                            image_output[batch_index][channel_index]
                        )
                    )
            return images, ranges if len(ranges) > 0 else None, loss

    def sample_and_save(
        self,
        milestone: Union[int, str] = None,
        save=True,
        progress_bar=False,
    ) -> Tuple[List[str], Optional[List[Tensor]], float]:
        image_filenames = []
        all_ranges = []
        total_sample_loss = 0.0
        num_batches = 0

        num_conditions = self.sample_dataset.conditions_per_plate_geometry
        num_steps = self.sample_dataset.num_steps

        if progress_bar:
            sample_dataloader_with_pb = tqdm(
                enumerate(self.sample_dataloader), desc="Sampling"
            )
        else:
            sample_dataloader_with_pb = enumerate(self.sample_dataloader)

        for batch_index, batch in sample_dataloader_with_pb:
            images, ranges, loss = self.sample(batch)

            if ranges is not None:
                all_ranges.append(ranges)

            total_sample_loss += loss
            num_batches += 1

            if not save:
                continue

            batch_outputs = (
                enumerate(zip(images, ranges)) if exists(ranges) else enumerate(images)
            )

            if progress_bar:
                batch_outputs = tqdm(batch_outputs, desc="Saving batch")

            for batch_output_index, output in batch_outputs:
                if exists(ranges):
                    image, range = output
                else:
                    image = output
                    range = None
                axis = "x" if batch_output_index % 2 == 0 else "y"
                index = batch_output_index // 2 + batch_index * self.sample_batch_size
                plate = (index // (num_conditions * num_steps)) + 1
                condition = (index % (num_conditions * num_steps)) // num_steps + 1
                step = (index % (num_conditions * num_steps)) % num_steps + 1

                if exists(milestone):
                    pathname = (
                        self.results_folder
                        / f"{milestone}"
                        / f"{plate}"
                        / f"{condition}"
                    )
                else:
                    pathname = self.results_folder / f"{plate}" / f"{condition}"

                pathname.mkdir(parents=True, exist_ok=True)

                plt.imsave(
                    str(pathname / f"sample_{axis}_{step}.png"),
                    torch.squeeze(image).clone().detach().cpu().numpy(),
                    cmap="Greys",
                    vmin=0,
                    vmax=255,
                )
                image_filenames.append(str(pathname / f"sample_{axis}_{step}.png"))

                if exists(range):
                    np.savetxt(
                        str(pathname / f"sample_{axis}_{step}.txt"),
                        range.clone().detach().cpu().numpy(),
                    )

        if num_batches != 0:
            total_sample_loss /= num_batches

        image_filenames = None if not save else image_filenames
        return (
            image_filenames,
            all_ranges if len(all_ranges) > 0 else None,
            total_sample_loss,
        )

    def train(self, wandb_inject_function=None):
        print(
            "Epoch Size: {} effective batches".format(
                (len(self.train_dataloader) / (self.num_gradient_accumulation_steps))
            )
        )
        print(
            "Number of Effective Epochs: {}".format(
                self.num_train_steps
                / (len(self.train_dataloader) / (self.num_gradient_accumulation_steps))
            )
        )
        with tqdm(
            initial=self.step.step,
            total=self.num_train_steps,
            disable=not self.accelerator.is_main_process,
        ) as progress_bar:
            lowest_sample_loss = float("inf")
            while self.step.step < self.num_train_steps:
                total_loss = 0.0
                # with tqdm(initial = 0, total = self.num_gradient_accumulation_steps, disable = not self.accelerator.is_main_process, leave=False) as inner_progress_bar:
                for _ in range(self.num_gradient_accumulation_steps):
                    batch: dict = next(self.train_yielder)

                    image_output, range_output = self.sample_model(batch)
                    loss = self.calculate_losses(
                        (
                            [image_output, *range_output]
                            if exists(range_output)
                            else [image_output]
                        ),
                        (
                            [
                                batch["displacement"],
                                batch["sign_displacement_range"],
                                batch["log_displacement_range"],
                            ]
                            if exists(range_output)
                            else [batch["displacement"]]
                        ),
                    )
                    loss = loss / self.num_gradient_accumulation_steps

                    total_loss += loss.item()
                    # inner_progress_bar.set_description(f'current batch loss: {total_loss:.4f}')
                    self.accelerator.backward(loss)
                    # inner_progress_bar.update(1)

                progress_bar.set_description(f"loss: {total_loss:.4f} ")

                logging.info(f"step: {self.step.step}, loss: {total_loss:.4f}")

                self.accelerator.wait_for_everyone()
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.max_gradient_norm
                )  # Gradient clipping

                self.optimizer.step()
                self.optimizer.zero_grad()

                self.accelerator.wait_for_everyone()

                self.step.step += 1

                if self.accelerator.is_main_process:
                    # self.ema.update()
                    total_sample_loss = None
                    image_filenames = None
                    ranges = None
                    milestone = None

                    if (
                        self.step.step != 0
                        and self.step.step % self.num_steps_per_milestone == 0
                    ):
                        # self.ema.ema_model.eval()

                        with torch.inference_mode():
                            milestone = self.step.step // self.num_steps_per_milestone
                            image_filenames, ranges, total_sample_loss = (
                                self.sample_and_save()
                            )
                            logging.info(f"sample loss: {total_sample_loss:.4f}")

                        if total_sample_loss < lowest_sample_loss:
                            lowest_sample_loss = total_sample_loss
                            self.save_checkpoint("best")
                        else:
                            self.save_checkpoint("latest")
                    elif (
                        self.step.step != 0
                        and self.step.step % self.num_steps_per_soft_milestone == 0
                    ):
                        with torch.inference_mode():
                            _, _, total_sample_loss = self.sample_and_save(save=False)
                            logging.info(f"sample loss: {total_sample_loss:.4f}")

                    if exists(wandb_inject_function):
                        wandb_inject_function(
                            self.step.step,
                            total_loss,
                            total_sample_loss,
                            image_filenames,
                            ranges,
                            milestone,
                        )

                progress_bar.update(1)

        self.accelerator.wait_for_everyone()
        self.save_checkpoint("final")
        self.accelerator.print("Training done!")
