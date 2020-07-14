#!/bin/bash

# sudo userdocker run -it -v /b_test:/b_test -v /ds:/ds dlcc/pytorch:19.09 /b_test/siddiqui/Repositories/ESD/examples/sweep.sh
export PATH="/home/siddiqui/anaconda3/bin/":$PATH
cd /b_test/siddiqui/Repositories/ESD/examples/
rm results.log
rm Plots/*
touch results.log

for mname in resnet18 resnet34 resnet50 resnet101 resnet152 vgg11 vgg11_bn vgg13 vgg13_bn vgg16 vgg16_bn vgg19 vgg19_bn; do
  for synthetic_data in 0 1; do
    extra_args=""
    if [[ $synthetic_data == 1 ]]; then
      extra_args="--synthetic_data"
    fi
    echo "Evaluating $mname model!"
    python example_imagenet.py --model_name $mname --data_path /ds/images/imagenet/ $extra_args | tee -a results.log
  done
done
