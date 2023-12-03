from model.diffusion import Trainer
from model.unet import UNet
from model.fdnunet import FDNUNet

import argparse

parser = argparse.ArgumentParser(description="Train model.")

parser.add_argument(
    "--data_dir", type=str, default="data/feadata2500", help="Data directory."
)
parser.add_argument(
    "--sample_data_dir",
    type=str,
    default="data/sample_1",
    help="Sample data directory.",
)
parser.add_argument(
    "--num_sample_conditions_per_plate",
    type=int,
    default=1,
    help="Number of sample conditions per plate.",
)
parser.add_argument(
    "--results_dir", type=str, default="results", help="Results directory."
)
parser.add_argument("--image_size", type=int, default=256, help="Image size.")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
parser.add_argument(
    "--num_gradient_accumulation_steps",
    type=int,
    default=16,
    help="Number of gradient accumulation steps.",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Checkpoint to load from (should be in results folder).",
    required=True,
)
parser.add_argument("--use_ema_model", action="store_true", help="Use EMA model.")
# parser.add_argument('--successive_sampling', action='store_true', help='Use successive sampling.')
parser.add_argument("--milestone", type=str, default="test", help="Milestone name.")

args = parser.parse_args()

# model = UNet(
#     input_dim=64,
#     num_channels=2,  # displacement (2)
#     num_condition_channels=4,  # constraints (1) + force (2) + geometry (1)
# )

model = FDNUNet(
    input_dim=64,
    num_channels=2, # geometry/displacement (2)
    # num_condition_channels=1, # geometry (1)
    num_auxiliary_condition_channels=3, # constraints (1) + force (2)
    num_stages=4
)

trainer = Trainer(
    model=model,
    dataset_folder=args.data_dir,
    sample_dataset_folder=args.sample_data_dir,
    num_sample_conditions_per_plate=args.num_sample_conditions_per_plate,
    num_gradient_accumulation_steps=args.num_gradient_accumulation_steps,
    dataset_image_size=args.image_size,
    train_batch_size=args.batch_size,
    results_folder=args.results_dir,
)

trainer.load_checkpoint(args.checkpoint)

# if args.successive_sampling:
#     trainer.successive_sample_and_save(milestone = args.milestone, use_ema_model=args.use_ema_model)
# else:
trainer.sample_and_save(milestone=args.milestone, use_ema_model=args.use_ema_model)
