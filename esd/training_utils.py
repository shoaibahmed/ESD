import torch
from tqdm import tqdm

from . import dist_utils
from . import logging_utils


def get_lr_policy(optimizer, lr_scheduler, param_dict):
    # if lr_scheduler is None:
    #     return None
    
    if lr_scheduler == "step_lr":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=param_dict["step_size"], 
                                                    gamma=param_dict["gamma"])
    elif lr_scheduler == "multistep_lr":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=param_dict["milestones"], 
                                                         gamma=param_dict["gamma"])
    elif lr_scheduler == "exponential_lr":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=param_dict["gamma"])
    elif lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=param_dict["train_epochs"])
    else:
        assert lr_scheduler == "reduce_lr_on_plateau", lr_scheduler
        raise RuntimeError("LR on Plateau is not supported since that requires the final loss.")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, mode='min', factor=param_dict["gamma"], 
                                                               patience=param_dict["patience"])
    return scheduler


def train(model, dataloader, device=None, progress_bar=False):
    optimizer = model.optimizer
    criterion = model.criterion

    model.train()
    pbar = tqdm(total=len(dataloader)) if progress_bar else None
    for input, target in dataloader:
        if device is not None:
            input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if pbar is not None: pbar.update(1)
    if pbar is not None: pbar.close


def evaluate(model, dataloader, device=None, progress_bar=False):
    criterion = model.criterion

    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(total=len(dataloader)) if progress_bar else None
        for input, target in dataloader:
            if device is not None:
                input, target = input.to(device), target.to(device)
            output = model(input)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += int((pred == target).sum())
            total += len(input)
            if pbar is not None: pbar.update(1)
        if pbar is not None: pbar.close
    
    # Collect statistics from different processes in case of a distributed job
    correct = int(dist_utils.reduce_tensor(torch.tensor(correct).cuda()))
    total = int(dist_utils.reduce_tensor(torch.tensor(total).cuda()))
    val_loss = float(dist_utils.reduce_tensor(torch.tensor(val_loss).cuda()))

    acc = float(correct) / total
    val_loss /= total
    log_dict = dict(correct=correct, total=total, val_loss=val_loss)
    logging_utils.log_info(f"Evaluation result | Average loss: {val_loss*dataloader.batch_size:.4f} | Accuracy: {correct}/{total} ({100.*acc:.2f}%)")
    return acc, log_dict
