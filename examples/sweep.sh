#!/bin/bash

for mname in resnet18 resnet34 resnet50 resnet101 resnet152 vgg11 vgg11_bn vgg13 vgg13_bn vgg16 vgg16_bn vgg19 vgg19_bn; do
  for synthetic_data in 0 1; do
    extra_args=""
    if [[ $synthetic_data == 1 ]]; then
      extra_args="--synthetic_data"
    fi
    python example_imagenet.py --model_name $mname $extra_args
  done
done
