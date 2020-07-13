import torch
from torch.utils.data import DataLoader, TensorDataset


class SynteticDataLoader:
    def __init__(
        self,
        fp16,
        batch_size,
        num_classes,
        num_channels,
        height,
        width,
        memory_format=torch.contiguous_format,
        device=torch.device("cuda"),
    ):
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


def get_syntetic_loader_same(
    batch_size,
    data_shape,
    num_classes,
    start_epoch=0,
    workers=None,
    _worker_init_fn=None,
    fp16=False,
    memory_format=torch.contiguous_format,
    device=torch.device("cuda"),
):
    assert isinstance(data_shape, list) or isinstance(data_shape, tuple)
    assert len(data_shape) == 3 and data_shape[0] in [1, 3]
    return SynteticDataLoader(fp16, batch_size, num_classes, data_shape[0], data_shape[1], data_shape[2],
                              memory_format=memory_format, device=device)


def get_syntetic_loader(
    batch_size,
    num_examples,
    data_shape,
    num_classes,
    start_epoch=0,
    workers=None,
    _worker_init_fn=None,
):
    assert isinstance(data_shape, list) or isinstance(data_shape, tuple)
    assert len(data_shape) == 3 and data_shape[0] in [1, 3]

    dataset = TensorDataset(
        torch.randn(num_examples, *data_shape),
        torch.randint(0, num_classes, (num_examples,))
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers if workers is not None else 8,
        shuffle=True,
        pin_memory=False
    )

    return loader
