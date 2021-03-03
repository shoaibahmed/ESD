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
rm "results_ts_dist_"$num_procs".log"
touch "results_ts_dist_"$num_procs".log"

optim_bs=1024
bs=$(($optim_bs/$num_procs))
lr=$(python -c "print($optim_bs/1000)")

echo "Evaluating time-series model using batch size of $bs (optimizer batch-size: $optim_bs) and LR of $lr!"
python multiproc.py --nproc_per_node $num_procs example_imagenet.py --model_name $mname --sync_bn \
       --data_path /ds/images/imagenet/ --batch_size $bs --train_epochs 100 --optimizer sgd --lr $lr \
       --momentum 0.9 --num_workers 4 --use_gpu_dl --debug --synthetic_data | tee -a "results_ts_dist_"$num_procs".log"
