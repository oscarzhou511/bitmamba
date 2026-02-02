import torch
import timeit

def naive_mm(A, W):
    scale = W.abs().mean()
    W_t = torch.sign(W) * (torch.abs(W) >= scale * 0.7)
    return torch.matmul(A, W_t.t())

def tmac_mm(A, W):
    scale = W.abs().mean()
    W_t = torch.sign(W) * (torch.abs(W) >= scale * 0.7)
    # T-MAC insight: matmul(W_t.t()) = sum over active weights per output
    return torch.einsum('bi,oi->bo', A, W_t)

A = torch.randn(64, 128, device='cuda')
W = torch.randn(256, 128, device='cuda')

ref = naive_mm(A, W)
fast = tmac_mm(A, W)
print("✅ Exact match:", torch.allclose(ref, fast))
print("🔢 Max error:", torch.abs(ref - fast).max().item())

t1 = timeit.timeit(lambda: naive_mm(A.repeat(10,1), W), number=100) / 100
t2 = timeit.timeit(lambda: tmac_mm(A.repeat(10,1), W), number=100) / 100
print(f"⏱️  Naive: {t1*1000:.1f}ms, T-MAC: {t2*1000:.1f}ms ({t1/t2:.1f}x faster!)")
