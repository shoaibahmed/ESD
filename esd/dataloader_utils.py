import torch
from torch.utils.data import DataLoader

import numpy as np
from functools import partial

from . import logging_utils
from .dataset_utils import TensorDataset, SubsetDataset


class PrefetchedWrapper(object):
    def prefetched_loader(loader, fp16=False, normalization=False):
        """
        Prefetching-based dataloader to be used on top of the default PyTorch dataloader.
        :param loader: default PyTorch dataloader to be wrapped.
        :param fp16: (optional) use FP-16 inputs instead of FP-32. If not defined, defaults to false.
        :param normalization: (optional) use input normalization from uint8 to float. This is required specifically
                                when using uint8 synthetic dataset. If not defined, defaults to false.
        :return iterator over the examples supplemented with prefetching
        """
        if normalization:
            raw_normalizer = torch.tensor([255.0]).cuda().view(1, 1, 1, 1)
            if fp16:
                raw_normalizer = raw_normalizer.half()

        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
                next_input = next_input.float()

                if normalization:
                    next_input = next_input.div_(raw_normalizer)

            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield input, target

    def __init__(self, dataloader, normalization=False):
        """
        Prefetching-based dataloader to be used on top of the default PyTorch dataloader.
        :param dataloader: default PyTorch dataloader to be wrapped.
        :param normalization: (optional) use input normalization from uint8 to float. This is required specifically
                                when using uint8 synthetic dataset. If not defined, defaults to false.
        :return object of the class
        """
        self.dataloader = dataloader
        self.epoch = 0
        self.batch_size = dataloader.batch_size
        self.normalization = normalization

    def __iter__(self):
        """
        :return new prefetched iterator
        """
        if self.dataloader.sampler is not None and isinstance(
            self.dataloader.sampler, torch.utils.data.distributed.DistributedSampler
        ):

            self.dataloader.sampler.set_epoch(self.epoch)
        self.epoch += 1
        return PrefetchedWrapper.prefetched_loader(self.dataloader)

    def __len__(self):
        """
        :return length of the dataset
        """
        return len(self.dataloader)


def fast_collate(memory_format, batch):
    """
    Advanced collate function which is specifically efficient for PrefetchedWrapper.
    :param memory_format: memory format used by the program. Usually, it is specified to be torch.contiguous_format.
    :param batch: examples in the batch provided in the form of tuples.
    :return batched output for both labels and targets
    """
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8).contiguous(
        memory_format=memory_format
    )
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)

    return tensor, targets


def fast_collate_tensors(memory_format, batch):
    """
    Advanced collate function which is specifically efficient when using TensorDataset.
    :param memory_format: memory format used by the program. Usually, it is specified to be torch.contiguous_format.
                    This option is not really required in this case, but kept for consistency with the other methods.
    :param batch: examples in the batch provided in the form of tuples.
    :return batched output for both labels and targets
    """
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    tensor = torch.stack(imgs, dim=0)
    return tensor, targets


def get_dataloader(dataset, batch_size, num_examples, workers=None, _worker_init_fn=None, world_size=1, gpu_dl=False):
    """
    Returns the dataloader to be used with the specified dataset.
    :param dataset: actual dataset object to be wrapped in the dataloader for training.
    :param batch_size: batch size to be used for training.
    :param num_examples: number of examples in the dataset. This option is required to take out a subset from the
                    original dataset which is usually larger.
    :param workers: (optional) number of workers to be used for training the model. If not defined, defaults to 8.
    :param _worker_init_fn: (optional) specifies the worker init function to be used. This is important for seeding purposes.
    :param world_size: (optional) specifies the number of processes launched. Useful for distributed training. If
                    not specified, defaults to 1.
    :param gpu_dl: (optional) use GPU dataloader which internally uses prefetching to speed up the training. If
                    not defined, defaults to false.
    """
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

    collate_fn = partial(fast_collate_tensors if synthetic_dataset else fast_collate, torch.contiguous_format)
    loader = DataLoader(
        dataset_small,
        batch_size=batch_size,
        num_workers=workers if workers is not None else 8,
        worker_init_fn=_worker_init_fn,
        shuffle=False,
        pin_memory=True if gpu_dl else False,
        sampler=dist_sampler,
        collate_fn=collate_fn,
    )

    if gpu_dl:
        assert "ToTensor" not in str(dataset.transform), dataset.transform
        normalization = not synthetic_dataset or dataset.dtype == "uint8"
        logging_utils.log_debug("Using GPU-based dataloader!")
        return PrefetchedWrapper(loader, normalization)

    return loader
