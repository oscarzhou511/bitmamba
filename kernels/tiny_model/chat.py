import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer

class TinyBitNet(nn.Module):
    def __init__(self, vocab_size=50257, d_model=128, n_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model),
                nn.LayerNorm(d_model)
            ) for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, vocab_size)
    
    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)

model = TinyBitNet().cuda()
model.load_state_dict(torch.load("tiny_bitnet_reg.pt", map_location='cuda'), strict=False)
model.eval()

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def generate(prompt, max_new=20):
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(tokens)
            next_token = torch.argmax(logits[0, -1], dim=-1).unsqueeze(0)
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    return tokenizer.decode(tokens[0], skip_special_tokens=True)

print("🤖 31% Ternary BitNet:")
print("France:", generate("The capital of France is"))
print("Math:", generate("2 + 2 ="))
print("Code:", generate("def hello():"))
