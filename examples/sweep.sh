#!/bin/bash

num_procs=1
if [[ "$#" -eq 1 ]]
then
  num_procs=$1
  echo "Starting "$num_procs"x distributed job!"
fi

# NV_GPU=3,4 sudo userdocker run -it -v /netscratch:/netscratch -v /ds:/ds dlcc/pytorch:19.09 /netscratch/siddiqui/Repositories/ESD/examples/sweep.sh 2
export PATH="/home/siddiqui/anaconda3/bin/":$PATH
cd /netscratch/siddiqui/Repositories/ESD/examples/
rm "results_dist_"$num_procs".log"
touch "results_dist_"$num_procs".log"

optim_bs=1024
bs=$(($optim_bs/$num_procs))
lr=$(python -c "print($optim_bs/1000)")

# for mname in resnet18 resnet34 resnet50 resnet101 resnet152 vgg11 vgg11_bn vgg13 vgg13_bn vgg16 vgg16_bn vgg19 vgg19_bn; do
for mname in resnet18 resnet34 resnet50; do
  for synthetic_data in 1 0; do
    extra_args=""
    if [[ $synthetic_data == 1 ]]; then
      extra_args="--synthetic_data"
    fi
    echo "Evaluating $mname model using batch size of $bs (optimizer batch-size: $optim_bs) and LR of $lr!"
    python multiproc.py --nproc_per_node $num_procs example_imagenet.py --model_name $mname --sync_bn \
           --data_path /ds/images/imagenet/ --batch_size $bs --train_epochs 100 --optimizer sgd --lr $lr \
           --momentum 0.9 --num_workers 4 --use_gpu_dl --debug $extra_args | tee -a "results_dist_"$num_procs".log"
    sleep 5  # To ensure that the port has bas been released for the subsequent process
  done
done
