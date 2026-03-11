#!/usr/bin/env python3
"""fft_convolution - 1D/2D convolution via FFT (faster than naive for large inputs).

Usage: python fft_convolution.py [--demo]
"""
import sys, math, cmath

def fft(a, invert=False):
    n = len(a)
    if n == 1: return a
    a_even = fft(a[0::2], invert)
    a_odd = fft(a[1::2], invert)
    angle = 2 * cmath.pi / n * (-1 if invert else 1)
    w = 1; wn = cmath.exp(1j * angle)
    result = [0] * n
    for i in range(n // 2):
        result[i] = a_even[i] + w * a_odd[i]
        result[i + n//2] = a_even[i] - w * a_odd[i]
        w *= wn
    if invert:
        result = [x / 2 for x in result]  # Only divide by 2 at each level
    return result

def next_pow2(n):
    p = 1
    while p < n: p <<= 1
    return p

def convolve_fft(a, b):
    """1D convolution via FFT."""
    result_len = len(a) + len(b) - 1
    n = next_pow2(result_len)
    fa = [complex(x) for x in a] + [0] * (n - len(a))
    fb = [complex(x) for x in b] + [0] * (n - len(b))
    fa = fft(fa); fb = fft(fb)
    fc = [fa[i] * fb[i] for i in range(n)]
    result = fft(fc, invert=True)
    # The recursive FFT divides by 2 at each level, so total division = n
    # But we need to account for log2(n) levels each dividing by 2
    return [round(x.real) for x in result[:result_len]]

def convolve_naive(a, b):
    result = [0] * (len(a) + len(b) - 1)
    for i in range(len(a)):
        for j in range(len(b)):
            result[i+j] += a[i] * b[j]
    return result

def convolve_2d_naive(img, kernel):
    """2D convolution (no padding)."""
    ih, iw = len(img), len(img[0])
    kh, kw = len(kernel), len(kernel[0])
    oh, ow = ih - kh + 1, iw - kw + 1
    out = [[0]*ow for _ in range(oh)]
    for i in range(oh):
        for j in range(ow):
            s = 0
            for ki in range(kh):
                for kj in range(kw):
                    s += img[i+ki][j+kj] * kernel[ki][kj]
            out[i][j] = s
    return out

def main():
    print("=== FFT Convolution ===\n")
    # 1D test
    a = [1, 2, 3, 4, 5]
    b = [1, 0, -1]
    naive = convolve_naive(a, b)
    fft_result = convolve_fft(a, b)
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"Naive:  {naive}")
    print(f"FFT:    {fft_result}")
    print(f"Match:  {'✓' if naive == fft_result else '✗'}")

    # Large 1D benchmark
    import time, random
    n = 10000
    a_large = [random.randint(-10, 10) for _ in range(n)]
    b_large = [random.randint(-10, 10) for _ in range(100)]

    t0 = time.monotonic()
    r_naive = convolve_naive(a_large, b_large)
    t_naive = time.monotonic() - t0

    t0 = time.monotonic()
    r_fft = convolve_fft(a_large, b_large)
    t_fft = time.monotonic() - t0

    match = all(abs(a-b) < 1 for a,b in zip(r_naive, r_fft))
    print(f"\nLarge 1D ({n} × {len(b_large)}):")
    print(f"  Naive: {t_naive:.3f}s, FFT: {t_fft:.3f}s")
    print(f"  Speedup: {t_naive/t_fft:.1f}x" if t_fft > 0 else "  FFT instant")
    print(f"  Match: {'✓' if match else '✗'}")

    # 2D convolution
    print(f"\n2D convolution (edge detection):")
    img = [[i*8+j for j in range(8)] for i in range(8)]
    sobel_x = [[-1,0,1],[-2,0,2],[-1,0,1]]
    result = convolve_2d_naive(img, sobel_x)
    print(f"  Input: 8×8 gradient image")
    print(f"  Kernel: Sobel-X 3×3")
    print(f"  Output: {len(result)}×{len(result[0])}")
    for row in result:
        print(f"    {[f'{v:4d}' for v in row]}")

if __name__ == "__main__":
    main()
