import torch


def solve(
    logits: torch.Tensor,
    p: torch.Tensor,
    seed: torch.Tensor,
    sampled_token: torch.Tensor,
    vocab_size: int,
):
    torch.manual_seed(int(seed.item()))
    logits = torch.nn.functional.softmax(logits.view(vocab_size, ), dim=0)
    logits, idx = torch.sort(logits, descending=True)
    cs = torch.cumsum(logits, dim=0)
    len_p = len(cs[cs < p])
    new_p = logits[:len_p+1]
    new_p /= new_p.sum() 
    sampled_token.copy_(idx[new_p.multinomial(1)])