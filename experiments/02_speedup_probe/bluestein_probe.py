"""Feasibility: prime-length FFT in fp16 via Bluestein (chirp-z).

cuFFT fp16 supports only power-of-2 lengths; the CR attention needs a length-p
(prime) FFT.  Bluestein converts a length-N DFT into chirp multiplications plus
a power-of-2 FFT, so the fp16 cuFFT kernel can be used.  The chirp phase is
computed from (n^2 mod 2N) so it stays representable in fp16 for large N.
"""
import torch

dev = "cuda"


def bluestein_fft(x, dim=-1, M=None):
    """Length-N DFT via Bluestein (chirp-z) using a power-of-2 FFT.

    Works in complex32 (fp16) or complex64 (fp32).
    """
    N = x.shape[dim]
    if M is None:
        M = 1
        while M < 2 * N - 1:
            M <<= 1
    dtype = x.dtype
    real_dtype = torch.float16 if dtype == torch.complex32 else torch.float32
    # phase arg = n^2 mod 2N (integer-exact), then angle = -pi * (n^2 mod 2N)/N
    n = torch.arange(N, device=x.device, dtype=torch.int64)
    phase = ((n * n) % (2 * N)).to(real_dtype) / N  # in [0, 2)
    w = torch.exp(-1j * torch.pi * phase).to(dtype)  # omega^{n^2}
    x = x.movedim(dim, -1)
    a = x * w
    # b carries the negative indices -(N-1)..-1 via the tail of the length-M array
    b = torch.zeros(M, dtype=dtype, device=x.device)
    b[:N] = torch.conj(w)
    b[M - N + 1:] = torch.conj(w[1:]).flip(0)
    A = torch.fft.fft(a, n=M, dim=-1)
    B = torch.fft.fft(b, n=M, dim=-1)
    y = torch.fft.ifft(A * B, n=M, dim=-1)[..., :N]
    y = y * w
    return y.movedim(-1, dim)


def main():
    torch.manual_seed(0)
    for N in (16, 17, 11, 101, 1331):
        x = torch.randn(4, N, 128, dtype=torch.complex64, device=dev)
        ref = torch.fft.fft(x, dim=1)

        # direct complex32 FFT (power-of-2 only)
        if N & (N - 1) == 0:
            y = torch.fft.fft(x.to(torch.complex32), dim=1)
            err = (y.to(torch.complex64) - ref).abs().max().item()
            print(f"N={N} pow2  c32 direct fft: err={err:.2e}")

        # Bluestein fp32
        yb32 = bluestein_fft(x, dim=1)
        err32 = (yb32 - ref).abs().max().item()
        # Bluestein fp16
        try:
            yb16 = bluestein_fft(x.to(torch.complex32), dim=1)
            err16 = (yb16.to(torch.complex64) - ref).abs().max().item()
            print(f"N={N} bluestein: fp32 err={err32:.2e}  fp16 err={err16:.2e}")
        except Exception as e:
            print(f"N={N} bluestein fp16: FAIL {type(e).__name__}: {str(e)[:70]}")


if __name__ == "__main__":
    main()
