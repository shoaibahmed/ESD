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
args = parser.parse_args()

assert args.synthetic_data or args.data_path is not None

# Create the model
generator = getattr(models, mname)
model = generator(pretrained=True)

data_shape = (3, 224, 224)

if not args.synthetic_data:
    raise NotImplementedError

# Optional to specify the training params and optimizer
training_params = {}
optimizer = None

esd = EmpiricalShatteringDimension(model=model,
                                   dataloader=None,
                                   synthetic_data=args.synthetic_data,
                                   data_shape=data_shape,
                                   num_classes=args.num_classes,
                                   optimizer=optimizer,
                                   training_params=training_params)
shattering_dim, log_dict = esd.evaluate()
plot_log(log_dict, output_file=None)
