import os
import argparse
import simplejson

import coloredlogs
import logging
import warnings

import torch
from torchvision import models, transforms, datasets

from torchmore import flex  # pip install -e git://github.com/tmbdev/torchmore.git#egg=torchmore

from esd import EmpiricalShatteringDimension
from esd.utils import plot_log


class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def create_classifier(num_timesteps, num_channels, num_labels, batch_norm=True, act_func="relu", verbose=False):
    assert act_func in ["relu", "prelu", "softplus", "gelu", "celu", "swish"], act_func
    assert batch_norm, "Batch-norm is usually always assumed to be turned on. Please use the --batch_norm flag!"
    input_shape = (1, num_channels, num_timesteps)

    conv_configs = [(256, True), (256, True), (256, True if num_timesteps > 50 else False), (256, True)]
    if num_timesteps >= 500:
        conv_configs += [(256, True)]
    conv_configs += [(256, True if num_timesteps > 500 else False)]

    fc_layers = [
        torch.nn.Flatten(),
        flex.Linear(num_labels)
    ]

    conv_layers = []
    for i, conv_config in enumerate(conv_configs):
        ks, pool = conv_config
        assert isinstance(ks, int) and isinstance(pool, bool)
        conv_layer = [flex.Conv1d(ks, 3, 1, 1)]
        if batch_norm:
            conv_layer += [torch.nn.BatchNorm1d(ks)]

        # Beta = 10 for SoftPlus is obtained from the following paper (https://arxiv.org/abs/2006.14536)
        activation_function = torch.nn.PReLU() if act_func == "prelu" else torch.nn.Softplus(beta=10) \
            if act_func == "softplus" else torch.nn.GELU() if act_func == "gelu" else torch.nn.CELU(alpha=1.0) \
            if act_func == "celu" else Swish() if act_func == "swish" else torch.nn.ReLU()
        conv_layer += [activation_function]

        if pool:
            conv_layer += [torch.nn.MaxPool1d(kernel_size=2)]
        conv_layers = conv_layers + conv_layer
    layers = conv_layers + fc_layers
    classifier = torch.nn.Sequential(*layers)
    flex.shape_inference(classifier, input_shape)

    return classifier


# Init parser
parser = argparse.ArgumentParser(description='Empirical Shattering Dimension for time-series data')
parser.add_argument('--num_timesteps', type=int, default=50, help='number of time-steps in the dataset')
parser.add_argument('--num_channels', type=int, default=3, help='number of channels in the dataset')

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

# assert args.synthetic_data or args.data_path is not None
assert args.synthetic_data, "Not tested for real time-series datasets as of yet!"

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
model = create_classifier().to(device)

# Wrap the model in the distributed wrapper
if args.distributed:
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                      output_device=args.local_rank, find_unused_parameters=True)

data_shape = (args.num_channels, args.num_timesteps)
dataset = None

# Optional to specify the training params and optimizer
training_params = {"optimizer": args.optimizer, "lr": args.lr, "momentum": args.momentum, "wd": args.wd, "bs": args.batch_size, \
    "train_epochs": args.train_epochs, "lr_scheduler": args.lr_scheduler, "lr_steps": args.lr_steps, "gamma": args.gamma, "step_size": 10}
optimizer = None
lr_scheduler = None

esd = EmpiricalShatteringDimension(model=model,
                                   dataset=dataset,
                                   data_shape=data_shape,
                                   num_classes=args.num_classes,
                                   min_examples=100000,
                                   max_examples=1000000,
                                   example_increment=100000,
                                   optimizer=optimizer,
                                   lr_scheduler=lr_scheduler,
                                   training_params=training_params,
                                   synthetic_dtype="uint8",
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
