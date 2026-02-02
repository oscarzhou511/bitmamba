import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Define Model Architecture (Must match training exactly) ---
class BitLinear158(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def absmean_quant(self, w):
        scale = w.abs().mean()
        mask = (w.abs() >= scale * 0.7).float()
        return torch.sign(w) * mask

    def forward(self, x):
        w_q = self.absmean_quant(self.weight)
        return F.linear(x, w_q, self.bias)

class BitFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = BitLinear158(d_model, d_ff)
        self.fc2 = BitLinear158(d_ff, d_model)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, d_model=1024, n_heads=16, d_ff=4096):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = BitFFN(d_model, d_ff)

    def forward(self, x, attn_mask=None):
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        h = self.ln2(x)
        x = x + self.ffn(h)
        return x

class BitNet300M(nn.Module):
    def __init__(self, vocab_size=50257, d_model=1024, n_layers=18, n_heads=16, d_ff=4096):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(1024, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_embed.weight

    def forward(self, input_ids):
        b, l = input_ids.shape
        pos = torch.arange(l, device=input_ids.device).long().unsqueeze(0).expand(b, l)
        x = self.tok_embed(input_ids) + self.pos_embed(pos)
        
        mask = torch.triu(torch.full((l, l), float("-inf"), device=input_ids.device), diagonal=1)
        for blk in self.blocks:
            x = blk(x, attn_mask=mask)
        return self.lm_head(self.ln_f(x))

# --- Load Model & Chat ---
print("🤖 Loading 300M Fast Model...")
model = BitNet300M().to(device)
try:
    # Load the FINAL checkpoint from the fast run
    model.load_state_dict(torch.load("bitnet_300m_fixed_5000.pt", map_location=device), strict=False)
    print("✅ Checkpoint loaded successfully.")
except Exception as e:
    print(f"❌ Error loading checkpoint: {e}")
    exit()

model.eval()

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def generate_smart(prompt, max_new=50, temp=0.7, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device).long()
    
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(inputs)
            next_logits = logits[:, -1, :]
            
            # Temperature
            probs = torch.softmax(next_logits / temp, dim=-1)
            
            # Top-p
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            probs[indices_to_remove] = 0
            probs = probs / probs.sum()
            
            # Sample
            next_token = torch.multinomial(probs, 1)
            
            inputs = torch.cat([inputs, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(inputs[0], skip_special_tokens=True)

print("\n💬 BITNET 300M CHAT (Wikitext Trained):")
print("-" * 40)
prompts = [
    "The United States is",
    "In the year 1999,",
    "The first computer was",
    "Football is played with",
    "To cook a validation split," # Tricky nonsense prompt
]

for p in prompts:
    print(f"Prompt: {p}")
    print(f"Output: {generate_smart(p)}")
    print("-" * 40)
