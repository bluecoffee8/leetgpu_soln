import torch


# logits, topk_weights, topk_indices are tensors on the GPU
def solve(
    logits: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    M: int,
    E: int,
    k: int,
):
    topk_weights[:], topk_indices[:] = torch.topk(logits, k, dim=1)
    topk_weights[:] = torch.nn.functional.softmax(topk_weights, dim=1)