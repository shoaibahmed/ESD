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
        self.dtype = inputs.dtype

    def __getitem__(self, index):
        input, target = self.inputs[index], self.targets[index]
        if self.transform is not None:
            input = self.transform(input)
        return input, target

    def __len__(self):
        return self.targets.size(0)


class SubsetDataset(Dataset):
    r"""Dataset useful for taking out subsets of the dataset.
    This is an updated version of the torch.utils.data.SubsetDataset as it assigns a random permutation of the labels
    and distributes it among the different processes in a distributed setting.
    Arguments:
        dataset: Dataset to select the subset from.
        num_examples (int): number of examples to be selected from the dataset.
    """

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
    """
    Returns a synthetic dataset which can be used to train the model.
    :param num_examples: maximum number of examples in the dataset.
    :param data_shape: shape of every example to be returned. Requires a list of either 3 elements [C, H, W] or
                        a list of two elements for sequential dataset [C, L].
    :param num_classes: number of classes in the dataset.
    :param dtype: (optional) Specifies the type of data to be used for training the model. Possible options are
                        ["uint8", "float"]. Uint8 is a reasonable option only for image dataset where each pixel
                        takes on the distinct value from 0-255. This significantly reduces the amount of memory
                        required to store the dataset. For using the library with non-image data, it's recommended
                        to use float as the datatype. If not defined, defaults to uint8.
    :param world_size: (optional) specifies the number of processes launched. Useful for distributed training. If
                    not specified, defaults to 1.
    :param gpu_dl: (optional) use GPU dataloader which internally uses prefetching to speed up the training. If
                    not defined, defaults to false. This is required here since the dataset will not normalize
                    data in this case.
    :return dataset object
    """
    assert isinstance(data_shape, list) or isinstance(data_shape, tuple)
    assert (len(data_shape) == 3 and data_shape[0] in [1, 3]) or len(data_shape) == 2
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
    """
    Replaces dataset targets with random targets. These random targets are also broadcasted to other processes from
    the main process when using distributed training.
    :param dataset: dataset whose targets are to be replaced.
    :param num_classes: number of classes in the dataset.
    :return dataset object with the new random targets
    """
    targets = torch.randint(0, num_classes, (len(dataset.targets),))
    dist_utils.broadcast_from_main(targets)
    dataset.targets = targets.numpy().tolist()
