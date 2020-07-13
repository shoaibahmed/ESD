#!/bin/python
import torch
from tqdm import tqdm


def train(model, dataloader, device=None):
    optimizer = model.optimizer
    criterion = model.criterion

    model.train()
    for batch_idx, (input, target) in tqdm(enumerate(dataloader)):
        if device is not None:
            input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def evaluate(model, dataloader, val=True, device=None):
    criterion = model.criterion

    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for input, target in tqdm(dataloader):
            if device is not None:
                input, target = input.to(device), target.to(device)
            output = model(input)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += int((pred == target).sum())
            total += len(input)
    acc = float(correct) / total
    val_loss /= total
    print(f"[ESD] {'Validation' if val else 'Test'} set: Average loss: {val_loss*args.val_batch_size:.4f}, Accuracy: {correct}/{total} ({100.*acc:.2f}%)")
    return acc
