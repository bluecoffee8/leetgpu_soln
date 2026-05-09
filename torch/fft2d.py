import torch


# signal, spectrum are tensors on the GPU
def solve(signal: torch.Tensor, spectrum: torch.Tensor, M: int, N: int):
    z = torch.fft.fft2(torch.complex(signal[0::2].view(M, N), signal[1::2].view(M, N)))
    spectrum[0::2] = z.real.view(-1)
    spectrum[1::2] = z.imag.view(-1)
