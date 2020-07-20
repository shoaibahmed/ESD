import torch
from torch.utils.data import DataLoader

import numpy as np
from functools import partial

from . import logging_utils
from .dataset_utils import TensorDataset, SubsetDataset


class PrefetchedWrapper(object):
    def prefetched_loader(loader, fp16=False, normalization=False):
        if normalization:
            mean = (
                torch.tensor([0.485, 0.456, 0.406])
                .cuda()
                .view(1, 3, 1, 1)
            ) * 255
            std = (
                torch.tensor([0.229, 0.224, 0.225])
                .cuda()
                .view(1, 3, 1, 1)
            ) * 255
            if fp16:
                mean = mean.half()
                std = std.half()
        else:
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
                    next_input = next_input.sub_(mean).div_(std)
                else:
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
        self.dataloader = dataloader
        self.epoch = 0
        self.batch_size = dataloader.batch_size
        self.normalization = normalization

    def __iter__(self):
        if self.dataloader.sampler is not None and isinstance(
            self.dataloader.sampler, torch.utils.data.distributed.DistributedSampler
        ):

            self.dataloader.sampler.set_epoch(self.epoch)
        self.epoch += 1
        return PrefetchedWrapper.prefetched_loader(self.dataloader)

    def __len__(self):
        return len(self.dataloader)


def fast_collate(memory_format, batch):
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


def get_dataloader(dataset, batch_size, num_examples, workers=None, _worker_init_fn=None, world_size=1, gpu_dl=False):
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
        pin_memory=True if gpu_dl else False,
        sampler=dist_sampler,
        collate_fn=partial(fast_collate, torch.contiguous_format),
    )

    if gpu_dl:
        assert "ToTensor" not in str(dataset.transform), dataset.transform
        logging_utils.log_debug("Using GPU-based dataloader!")
        return PrefetchedWrapper(loader)

    return loader
