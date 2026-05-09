import torch


# signal and spectrum are device pointers
def solve(signal: torch.Tensor, spectrum: torch.Tensor, N: int):
    z = torch.fft.fft(torch.complex(signal[0::2], signal[1::2]))
    spectrum[0::2] = z.real.view(-1)
    spectrum[1::2] = z.imag.view(-1)