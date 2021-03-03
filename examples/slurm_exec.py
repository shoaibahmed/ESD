import os
import numpy as np

# python multiproc.py --nproc_per_node $num_procs example_imagenet.py --model_name $mname --sync_bn \
#            --data_path /ds/images/imagenet/ --batch_size $bs --train_epochs 100 --optimizer sgd --lr $lr \
#            --momentum 0.9 --num_workers 4 --use_gpu_dl --debug --synthetic_data | tee -a "results_dist_"$num_procs".log"


if __name__ == "__main__":
    dataset = "yfcc100m"
    synthetic_data = False
    sync_bn = False
    model_name = "resnet18"
    assert dataset in ["yfcc100m", "imagenet"]
    if dataset == "imagenet":
        data_path = "/ds/images/imagenet/"
    else:
        data_path = "/ds/images/YFCC10m_public/data/images/"
    
    # partitions = ["batch", "RTX3090", "RTX2080Ti", "GTX1080Ti", "RTX6000", "A100", "V100-32GB", "V100-16GB"]
    partitions = ["RTX3090"]
    optimizer = "adam"
    root_dir = "/netscratch/siddiqui/Repositories/ESD/examples/"
    log_dir = os.path.join(root_dir, f"logs_{dataset}")
    if not os.path.exists(log_dir):
        print("Creating log dir:", log_dir)
        os.mkdir(log_dir)
    
    job_name = f"{dataset}{'_syn' if synthetic_data else ''}_{model_name}"
    partition = np.random.choice(partitions)
    num_tasks = 4
    num_workers = 6
    mem = 256
    image_size = 32
    epochs = "25"
    optimizer_batch_size = 1024
    assert optimizer_batch_size % num_tasks == 0
    batch_size = optimizer_batch_size // num_tasks
    print(f"Using batch size of: {batch_size} | Optimizer batch size: {optimizer_batch_size}")
    
    slurm_cmd = ["srun", "-p", partition, "-K", f"--ntasks-per-node={num_tasks}", "--gpus-per-task=1", f"--cpus-per-gpu={num_workers}", 
                 f"--mem={mem}G", "--kill-on-bad-exit", "--job-name", job_name, "--nice=100", 
                 "--container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui", 
                 "--container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_20.08-py3.sqsh", 
                 "--container-workdir=`pwd`", "--container-mount-home", "--export='NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5'"]
    
    cmd = [
        "/opt/conda/bin/python", os.path.join(root_dir, "example_imagenet.py"),
        "--model_name", model_name,
        "--data_path", data_path,
        "--train_epochs", epochs,
        "--optimizer", optimizer,
        "--lr", "1e-3",
        "--momentum", "0.9",
        "--batch_size", f"{batch_size}",
        "--num_workers", f"{num_workers}",
        "--use_gpu_dl", "--debug",
        "--image_size", f"{image_size}",
    ]
    if synthetic_data:
        cmd += ["--synthetic_data"]
    if sync_bn:
        cmd += ["--sync_bn"]
    
    log_file = os.path.join(log_dir, job_name + ".log")
    output_redir = [">", log_file, "2>&1 &"]
    final_cmd = " ".join(slurm_cmd) + " " + " ".join(cmd) + " " + " ".join(output_redir)
    
    print(f"\n>>> {final_cmd}")
    os.system(final_cmd)
