import os
import argparse
import simplejson

import coloredlogs
import logging
import warnings

import torch
from torchvision import models, transforms, datasets

from esd import EmpiricalShatteringDimension
from esd.utils import plot_log


# Init parser
parser = argparse.ArgumentParser(description='Empirical Shattering Dimension')
parser.add_argument('--model_name', type=str, default="resnet50", help='model name')
parser.add_argument('--data_path', type=str, default=None, help='path to the train dataset')
parser.add_argument('--synthetic_data', action='store_true', help='use synthetic data')
parser.add_argument('--num_classes', type=int, default=1000, help='number of classes to evaluate')

parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--train_epochs', type=int, default=50, help='number of training epochs')

parser.add_argument('--optimizer', choices=["adam", "sgd"], default="adam", help='optimizer to be used')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum (only required for SGD)')

parser.add_argument('--lr_scheduler', choices=["none", "step", "cosine"], default="cosine", help='LR scheduler to be used')
parser.add_argument('--lr_steps', type=str, default=None, help='LR steps (only required for step LR scheduler)')
parser.add_argument('--gamma', type=float, default=0.1, help='gamma')

parser.add_argument('--plots_dir', type=str, default="Plots/", help='directory to store the complete output plots')
parser.add_argument('--debug', action='store_true', help='enable debug logging')

parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--num_gpus", type=int, default=1)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument('--sync_bn', action='store_true')
parser.add_argument('--use_gpu_dl', action='store_true', help='use GPU dataloader which might be more efficient in practice')

args = parser.parse_args()

assert args.synthetic_data or args.data_path is not None

# Setup logging
logger = logging.getLogger('ESD')  # Create a logger object
coloredlogs.install(level='DEBUG' if args.debug else 'INFO')
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

# Initialize the distributed environment
args.distributed = args.num_gpus > 1
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1

args.gpu = args.local_rank
args.world_size = 1
if args.distributed:
    assert args.local_rank >= 0
    torch.cuda.set_device(args.gpu)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    args.world_size = torch.distributed.get_world_size()
    args.num_gpus = torch.cuda.device_count()
    print(f"Initializing distributed environment with {args.num_gpus} GPUs...")
args.optimizer_batch_size = args.batch_size * args.world_size

main_proc = not torch.distributed.is_initialized() or args.local_rank == 0  # One main proc per node
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the model
generator = getattr(models, args.model_name)
model = generator(pretrained=False).to(device)

# Wrap the model in the distributed wrapper
if args.distributed:
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                      output_device=args.local_rank, find_unused_parameters=True)

data_shape = (3, 224, 224)
dataset = None
if not args.synthetic_data:
    print("Loading dataset...")
    traindir = os.path.join(args.data_path, "train")
    dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(
            [transforms.Resize((224, 224))] + ([] if args.use_gpu_dl else [transforms.ToTensor()])
        ),
    )

# Optional to specify the training params and optimizer
training_params = {"optimizer": args.optimizer, "lr": args.lr, "momentum": args.momentum, "wd": args.wd, "bs": args.batch_size, \
    "train_epochs": args.train_epochs, "lr_scheduler": args.lr_scheduler, "lr_steps": args.lr_steps, "gamma": args.gamma, "step_size": 10}
optimizer = None
lr_scheduler = None

esd = EmpiricalShatteringDimension(model=model,
                                   dataset=dataset,
                                   data_shape=data_shape,
                                   num_classes=args.num_classes,
                                   optimizer=optimizer,
                                   lr_scheduler=lr_scheduler,
                                   training_params=training_params,
                                   synthetic_dtype="uint8",
                                   max_examples=1000000,
                                   example_increment=100000,
                                   seed=args.seed,
                                   workers=args.num_workers,
                                   gpu_dl=args.use_gpu_dl)
shattering_dim, log_dict = esd.evaluate(acc_thresh=0.8, termination_thresh=0.1)

if main_proc:
    if not os.path.exists(args.plots_dir):
        os.mkdir(args.plots_dir)
    plot_log(log_dict, output_file=os.path.join(args.plots_dir, f"{args.model_name}{'_syn' if args.synthetic_data else ''}.png"))

    output_dict = {}
    output_dict["esd"] = shattering_dim
    output_dict["esd_log"] = log_dict
    output_dict["args"] = args.__dict__
    print(simplejson.dumps(output_dict))
