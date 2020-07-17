import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset  # SubsetRandomSampler, TensorDataset

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


class SynteticDataLoader:
    def __init__(self, fp16, batch_size, num_classes, num_channels, height, width, 
                 memory_format=torch.contiguous_format, device=torch.device("cuda")):
        input_data = (
            torch.empty(batch_size, num_channels, height, width).contiguous(memory_format=memory_format).to(device).uniform_(0, 1.0)
        )
        input_target = torch.randint(0, num_classes, (batch_size,)).to(device)
        if fp16:
            assert device.type == 'cuda'
            input_data = input_data.half()

        self.input_data = input_data
        self.input_target = input_target

    def __iter__(self):
        while True:
            yield self.input_data, self.input_target


def get_syntetic_loader_gpu(batch_size, data_shape, num_classes, fp16=False,
                            memory_format=torch.contiguous_format, device=torch.device("cuda")):
    assert isinstance(data_shape, list) or isinstance(data_shape, tuple)
    assert len(data_shape) == 3 and data_shape[0] in [1, 3]
    return SynteticDataLoader(fp16, batch_size, num_classes, data_shape[0], data_shape[1], data_shape[2],
                              memory_format=memory_format, device=device)


def get_syntetic_dataset(num_examples, data_shape, num_classes, dtype="float", world_size=1):
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
        transform=None if dtype == "float" else ToTensor(),
    )
    return dataset


def replace_dataset_targets(dataset, num_classes):
    targets = torch.randint(0, num_classes, (len(dataset.targets),))
    dist_utils.broadcast_from_main(targets)
    dataset.targets = targets.numpy().tolist()


def get_dataloader(dataset, batch_size, num_examples, workers=None, _worker_init_fn=None, world_size=1):
    assert num_examples <= len(dataset)

    synthetic_dataset = isinstance(dataset, TensorDataset)
    if synthetic_dataset:
        # Number of examples to be used should be uniformly distributed across processes as there is no distributed sampler
        assert num_examples % world_size == 0
        num_examples = num_examples // world_size
    
    dataset_small = SubsetDataset(dataset, num_examples)  # Sample only num_examples from the dataset
    dist_sampler = None
    if torch.distributed.is_initialized() and not isinstance(dataset, TensorDataset):  # Tensor dataset has already splitted the examples over processes
        logging_utils.log_debug("Using distributed sampler for external dataset!")
        dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset_small)

    loader = DataLoader(
        dataset_small,
        batch_size=batch_size,
        num_workers=workers if workers is not None else 8,
        worker_init_fn=_worker_init_fn,
        shuffle=False,
        pin_memory=False,
        sampler=dist_sampler
    )
    return loader
