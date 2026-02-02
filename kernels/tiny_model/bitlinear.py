class BitLinear(nn.Module):
    def __init__(self, in_features, out_features, ternary_lambda=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.ternary_lambda = ternary_lambda
    
    def ternary_loss(self, w):
        scale = w.abs().mean()
        q_error = torch.mean((w - torch.round(w / scale).clamp(-1, 1) * scale)**2)
        return q_error
    
    def forward(self, x):
        w_q = torch.sign(self.weight) * (torch.abs(self.weight) >= torch.abs(self.weight).mean())
        return F.linear(x, w_q, self.bias)
    
    @torch.enable_grad()
    def backward_hook(self, grad_output, grad_input):
        # STE: straight-through estimator
        return grad_input  # pretend gradients flowed through quantized weights
