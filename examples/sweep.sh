#!/bin/bash

num_procs=1
if [[ "$#" -eq 1 ]]
then
  num_procs=$1
  echo "Starting "$num_procs"x distributed job!"
fi

# NV_GPU=2,3 sudo userdocker run -it -v /netscratch:/netscratch -v /ds:/ds dlcc/pytorch:19.09 /netscratch/siddiqui/Repositories/ESD/examples/sweep.sh 2
export PATH="/home/siddiqui/anaconda3/bin/":$PATH
cd /netscratch/siddiqui/Repositories/ESD/examples/
rm results_dist.log
touch results_dist.log

optim_bs=256
bs=$(($optim_bs/$num_procs))

# for mname in resnet18 resnet34 resnet50 resnet101 resnet152 vgg11 vgg11_bn vgg13 vgg13_bn vgg16 vgg16_bn vgg19 vgg19_bn; do
for mname in resnet18 resnet34 resnet50; do
  for synthetic_data in 0 1; do
    extra_args=""
    if [[ $synthetic_data == 1 ]]; then
      extra_args="--synthetic_data"
    fi
    echo "Evaluating $mname model!"
    python multiproc.py --nproc_per_node $num_procs example_imagenet.py --model_name $mname --sync_bn \
           --data_path /ds/images/imagenet/ --batch_size $bs --lr 1e-3 --num_workers 4 $extra_args | tee -a results_dist.log
  done
done
