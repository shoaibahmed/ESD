import argparse

import torch
from torchvision import models

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

args = parser.parse_args()

assert args.synthetic_data or args.data_path is not None

# Create the model
generator = getattr(models, args.model_name)
model = generator(pretrained=False)

data_shape = (3, 224, 224)

if not args.synthetic_data:
    raise NotImplementedError

# Optional to specify the training params and optimizer
training_params = {"optimizer": args.optimizer, "lr": args.lr, "wd": args.wd, "bs": args.batch_size, "train_epochs": args.train_epochs}
optimizer = None

esd = EmpiricalShatteringDimension(model=model,
                                   dataset=None,
                                   synthetic_data=args.synthetic_data,
                                   data_shape=data_shape,
                                   num_classes=args.num_classes,
                                   optimizer=optimizer,
                                   training_params=training_params)
shattering_dim, log_dict = esd.evaluate(acc_thresh=0.8)
plot_log(log_dict, output_file=None)
