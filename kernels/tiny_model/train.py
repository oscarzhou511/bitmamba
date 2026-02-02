import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from transformers import GPT2Tokenizer
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling

class BitLinear(nn.Module):
    def __init__(self, in_features, out_features, ternary_lambda=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.ternary_lambda = ternary_lambda
    
    def ternary_loss(self, w):
        scale = w.abs().mean()
        q_error = torch.mean((w - torch.round(w / scale).clamp(-1, 1) * scale)**2)
        return q_error
    
    def absmean_quant(self, x):
        scale = x.abs().mean()
        return torch.sign(x) * (torch.abs(x) >= scale * 0.7).float()
    
    def forward(self, x):
        w_q = self.absmean_quant(self.weight)
        return F.linear(x, w_q, self.bias)

class TinyBitNet(nn.Module):
    def __init__(self, vocab_size=50257, d_model=128, n_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.Sequential(
                BitLinear(d_model, d_model * 4),
                nn.GELU(),
                BitLinear(d_model * 4, d_model),
                nn.LayerNorm(d_model)
            ) for _ in range(n_layers)
        ])
        self.head = BitLinear(d_model, vocab_size)
    
    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)

# Data
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").select(range(1000))
tokenized = dataset.map(lambda x: tokenizer(x["text"], truncation=True, max_length=128), batched=True, remove_columns=dataset.column_names)
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

model = TinyBitNet().cuda()
optimizer = optim.AdamW(model.parameters(), lr=6e-4)
dataloader = DataLoader(tokenized, batch_size=8, collate_fn=collator, shuffle=True)

print("🚀 Training with ternary reg...")
for epoch in range(2):
    total_loss = 0
    tern_total = 0
    steps = 0
    for i, batch in enumerate(dataloader):
        if steps > 80: break
        input_ids = batch["input_ids"].cuda()
        labels = input_ids.clone()
        logits = model(input_ids)
        ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        
        # Ternary regularization
        tern_loss = sum(m.ternary_loss(m.weight) for m in model.modules() if hasattr(m, 'ternary_loss'))
        loss = ce_loss + 0.1 * tern_loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += ce_loss.item()
        tern_total += tern_loss.item()
        steps += 1
        if i % 10 == 0:
            print(f"Epoch {epoch}, Step {i}, CE: {ce_loss:.3f}, Tern: {tern_loss:.3f}")
    
    print(f"✅ Epoch {epoch}: Avg CE {total_loss/steps:.3f}, Avg Tern {tern_total/steps:.3f}")

torch.save(model.state_dict(), "tiny_bitnet_reg.pt")

def check_ternary_fixed(model):
    ternary_pct = []
    for name, m in model.named_modules():
        if hasattr(m, 'absmean_quant'):
            w = m.weight.detach()
            w_q = m.absmean_quant(w)
            pct = 100 * torch.isclose(w, w_q, atol=0.01).float().mean().item()
            ternary_pct.append(pct)
            print(f"  {name}: {pct:.1f}%")
    print(f"📊 AVG: {sum(ternary_pct)/len(ternary_pct):.1f}%")

check_ternary_fixed(model)

print("🎉 Done!")
