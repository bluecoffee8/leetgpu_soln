import torch
import torch.nn.functional as F

# x, output, weights are tensors on the GPU
def solve(x: torch.Tensor, output: torch.Tensor, weights: torch.Tensor, seq_len: int):
    gamma1 = weights[:768]
    beta1 = weights[768:768+768]
    Wqkv = weights[1536:1536+1769472].view(768,2304)
    bqkv = weights[1771008:1773312].view(1, 2304)
    Wattn = weights[1773312:2363136].view(768,768)
    battn = weights[2363136:2363904].view(1, 768)
    gamma2 = weights[2363904:2364672]
    beta2 = weights[2364672:2365440]
    Wfc = weights[2365440:4724736].view(768,3072)
    bfc = weights[4724736:4727808].view(1,3072)
    Wproj = weights[4727808:7087104].view(3072,768)
    bproj = weights[7087104:7087104+768].view(1, 768)

    x = x.view(seq_len, 768)
    dk = 64

    xln = F.layer_norm(x, normalized_shape=[768], weight=gamma1, bias=beta1)
    Q, K, V = torch.split(xln @ Wqkv + bqkv, 768, dim=-1)

    Q = Q.view(seq_len, -1, dk)
    K = K.view(seq_len, -1, dk)
    V = V.view(seq_len, -1, dk)

    score = torch.einsum("mhd,nhd->mhn", Q, K) * (dk**-0.5)

    head = torch.einsum("mhn,nhl->mhl", torch.softmax(score, dim=-1), V).reshape(seq_len, -1)

    mh = head @ Wattn + battn

    x2 = x + mh

    xln2 =  F.layer_norm(x2, normalized_shape=[768], weight=gamma2, bias=beta2)

    ff = F.gelu(xln2 @ Wfc + bfc) @ Wproj + bproj
    output.copy_(ff + x2)