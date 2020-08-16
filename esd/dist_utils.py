import torch
import pickle


def is_main_proc():
    """
    :return Returns a bool indicating whether the process is the main process in the process tree when using
    distributed environment.
    """
    main_proc = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    return main_proc


def broadcast_from_main(tensor, is_tensor=True):
    """
    Broadcasts the provided tensor from the main process to all the other processes in the process tree when using
    distributed environment.
    :param tensor: tensor to be broadcasted.
    :param is_tensor: (optional) boolean value indicating whether the provided value is a tensor. Pickles the objects in the
    other case to prepare them for the transfer. If not defined, defaults to true.
    :return Returns the received tensor.
    """
    if not torch.distributed.is_initialized():
        return tensor
    
    if is_tensor:
        tensor = tensor.cuda()
    else:
        # Serialize data to a Tensor
        buffer = pickle.dumps(tensor)
        storage = torch.ByteStorage.from_buffer(buffer)
        tensor = torch.ByteTensor(storage).cuda()
    
    torch.distributed.broadcast(tensor, src=0)
    assert (reduce_tensor(tensor, average=True) - tensor <= 1e-6).all()
    return tensor


def reduce_tensor(tensor, average=False):
    """
    Reduces the provided tensor when using distributed environment.
    :param tensor: tensor to be broadcasted.
    :param average: (optional) boolean value indicating whether to average the values of the tensor from the different processes.
    Just sums up the values otherwise. If not defined, defaults to false.
    :return Returns the received tensor.
    """
    if not torch.distributed.is_initialized():
        return tensor
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    if average:
        rt /= torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    return rt


def gather_tensor(tensor):
    """
    Gathers the provided tensor when using distributed environment.
    :param tensor: tensor to be gathered.
    :return Returns the received tensor.
    """
    if not torch.distributed.is_initialized():
        return tensor
    tensor_list = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensor_list, tensor)
    tensor = torch.cat(tensor_list, dim=0)
    return tensor
