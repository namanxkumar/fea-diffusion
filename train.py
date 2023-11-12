from model.diffusion import Trainer
from model.unet import UNet

import argparse

parser = argparse.ArgumentParser(description='Train model.')

parser.add_argument('--data_dir', type=str, default='data', help='Data directory.')
parser.add_argument('--sample_data_dir', type=str, default='sample_data', help='Sample data directory.')
parser.add_argument('--results_dir', type=str, default='results', help='Results directory.')
parser.add_argument('--image_size', type=int, default=256, help='Image size.')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
parser.add_argument('--num_steps', type=int, default=100, help='Number of steps.')
parser.add_argument('--num_steps_per_milestone', type=int, default=1000, help='Number of steps per milestone.')
parser.add_argument('--ema_steps_per_milestone', type=int, default=10, help='EMA steps per milestone.')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate.')

args = parser.parse_args()

model = UNet(
    input_dim=64,
    num_channels=2, # geometry/displacement (2)
    num_condition_channels=4, # constraints (1) + force (2) + geometry (1)
)

trainer = Trainer(
    model=model,
    dataset_folder=args.data_dir,
    use_dataset_augmentation=True,
    sample_dataset_folder=args.sample_data_dir,
    num_sample_conditions_per_plate=1,
    num_gradient_accumulation_steps=1,
    dataset_image_size=args.image_size,
    train_batch_size=args.batch_size,
    train_learning_rate=args.learning_rate,
    num_train_steps=args.num_steps,
    num_steps_per_milestone=args.num_steps_per_milestone,
    ema_steps_per_milestone=args.ema_steps_per_milestone,
    results_folder=args.results_dir,
    use_batch_split_over_devices=True,
)

trainer.train()