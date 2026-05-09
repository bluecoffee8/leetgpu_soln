import torch
import torch.nn.functional as F

# x, output, weights, cos, sin are tensors on the GPU
def solve(
    x: torch.Tensor,
    output: torch.Tensor,
    weights: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seq_len: int,
):
    w1 = weights[0:0+512]
    Wq = weights[512:512+262144].view(512, 512)
    Wk = weights[262656:262656+65536].view(128, 512)
    Wv = weights[328192:328192+65536].view(128, 512)
    Wo = weights[393728:393728+262144].view(512, 512)
    w2 = weights[655872:655872+512]
    Wgate = weights[656384:656384+720896].view(1408, 512)
    Wup = weights[1377280:1377280+720896].view(1408, 512)
    Wdown = weights[2098176:2098176+720896].view(512, 1408)

    x = x.reshape(-1, 512)
    h = F.rms_norm(x, (512, ), eps=1e-5) * w1
    Q = (h @ Wq.T).reshape(-1, 8, 64)
    K = (h @ Wk.T).reshape(-1, 2, 64)
    V = (h @ Wv.T).reshape(-1, 2, 64)

    cos = cos[:, None, :]
    sin = sin[:, None, :]
    q1 = Q[..., :32]
    q2 = Q[..., 32:]
    k1 = K[..., :32]
    k2 = K[..., 32:]
    Q = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    K = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)

    K = K.repeat_interleave(4, dim=1)
    V = V.repeat_interleave(4, dim=1)

    scores = torch.bmm(Q.permute(1, 0, 2), K.permute(1, 2, 0)) / (64 ** 0.5)
    masked = torch.tril(scores) + torch.triu(torch.full((8, seq_len, seq_len), float("-inf"), device=Q.device), diagonal=1)
    h = torch.bmm(torch.softmax(masked, dim=-1), V.permute(1, 0, 2)).permute(1, 0, 2).reshape(-1, 512) # (8, seq_len, 64)
    x = (h @ Wo.T) + x

    h = F.rms_norm(x, (512, ), eps=1e-5) * w2
    output[:] = x + (F.silu(h @ Wgate.T) * (h @ Wup.T)) @ Wdown.T
