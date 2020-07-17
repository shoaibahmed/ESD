import torch
from tqdm import tqdm

from . import dist_utils


def train(model, dataloader, device=None, logging=False):
    optimizer = model.optimizer
    criterion = model.criterion

    model.train()
    pbar = tqdm(total=len(dataloader)) if logging else None
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


def evaluate(model, dataloader, device=None, logging=False):
    criterion = model.criterion

    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(total=len(dataloader)) if logging else None
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
    print(f"[ESD] Evaluation result | Average loss: {val_loss*dataloader.batch_size:.4f} | Accuracy: {correct}/{total} ({100.*acc:.2f}%)")
    return acc
