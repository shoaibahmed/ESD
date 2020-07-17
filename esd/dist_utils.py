import torch
import pickle


def broadcast_from_main(tensor, is_tensor=True):
    if not torch.distributed.is_initialized():
        return tensor
    rank = torch.distributed.get_rank()
    
    if is_tensor:
        tensor = tensor.cuda()
    else:
        # Serialize data to a Tensor
        buffer = pickle.dumps(tensor)
        storage = torch.ByteStorage.from_buffer(buffer)
        tensor = torch.ByteTensor(storage).cuda()
    
    if rank == 0:
        torch.distributed.broadcast(tensor, src=0)
        # assert (reduce_tensor(tensor, average=True) - tensor <= 1e-6).all()
    return tensor


def reduce_tensor(tensor, average=False):
    if not torch.distributed.is_initialized():
        return tensor
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    if average:
        rt /= torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    return rt


def gather_tensor(tensor):
    if not torch.distributed.is_initialized():
        return tensor
    tensor_list = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensor_list, tensor)
    tensor = torch.cat(tensor_list, dim=0)
    return tensor
