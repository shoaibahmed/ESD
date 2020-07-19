import torch
import numpy as np
from torch.utils.data import Dataset  # SubsetRandomSampler, TensorDataset

from . import dist_utils
from . import logging_utils


class ToTensor(object):
    """Converts a uint8 tensor of range [0, 255] to float with a range of [0, 1]"""
    def __call__(self, x):
        """
        Args:
            x: uint8 Tensor with range [0, 255]
        Returns:
            Tensor: converted float Tensor with range [0, 1]
        """
        x = x.float() / 255.
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'


class TensorDataset(Dataset):
    r"""Dataset wrapping tensors (updated version of torch.utils.data.TensorDataset which supports transforms)
    Each sample will be retrieved by indexing tensors along the first dimension.
    Arguments:
        inputs (Tensor): input data which will be fed in to the network
        targets (Tensor): targets to be used for training
    """

    def __init__(self, inputs, targets, transform=None):
        assert inputs.size(0) == targets.size(0)
        self.inputs = inputs
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        input, target = self.inputs[index], self.targets[index]
        if self.transform is not None:
            input = self.transform(input)
        return input, target

    def __len__(self):
        return self.targets.size(0)


class SubsetDataset(Dataset):
    def __init__(self, dataset, num_examples):
        assert len(dataset) >= num_examples
        self.dataset = dataset
        self.num_examples = num_examples
        
        random_example_idx = torch.from_numpy(np.random.choice(np.arange(len(dataset)), num_examples))
        dist_utils.broadcast_from_main(random_example_idx)  # Send to all other processes
        self.random_example_idx = random_example_idx.numpy().tolist()
        assert len(self.random_example_idx) == self.num_examples

    def __getitem__(self, index):
        new_idx = self.random_example_idx[index]
        return self.dataset[new_idx]

    def __len__(self):
        return self.num_examples


def get_syntetic_dataset(num_examples, data_shape, num_classes, dtype="float", world_size=1, gpu_dl=False):
    assert isinstance(data_shape, list) or isinstance(data_shape, tuple)
    assert len(data_shape) == 3 and data_shape[0] in [1, 3]
    assert dtype in ["uint8", "float"]
    assert world_size >= 1
    assert world_size == 1 or torch.distributed.is_initialized()

    # Number of examples are uniformly distributed for each process (doesn't require distributed sampler)
    assert num_examples % world_size == 0
    num_examples = num_examples // world_size
    logging_utils.log_debug(f"Creating synthetic dataset of size {num_examples} for each process ({world_size})!")

    tensor_shape = (num_examples, *data_shape)
    dataset = TensorDataset(
        torch.randn(*tensor_shape) if dtype == "float" else torch.randint(0, 256, tensor_shape, dtype=torch.uint8),
        torch.randint(0, num_classes, (num_examples,)),
        transform=None if dtype == "float" or gpu_dl else ToTensor(),
    )
    return dataset


def replace_dataset_targets(dataset, num_classes):
    targets = torch.randint(0, num_classes, (len(dataset.targets),))
    dist_utils.broadcast_from_main(targets)
    dataset.targets = targets.numpy().tolist()
