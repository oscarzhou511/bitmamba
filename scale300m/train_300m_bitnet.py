#!/usr/bin/env python3
import os
os.environ["TORCH_CUDA_MEMPOOL_DISABLE_RESERVOIR"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
from torch.amp import autocast, GradScaler

device = "cuda" if torch.cuda.is_available() else "cpu"

class BitLinear158(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def absmean_quant(self, w):
        scale = w.abs().mean()
        mask = (w.abs() >= scale * 0.7).float()
        return torch.sign(w) * mask

    def forward(self, x):
        w_q = self.absmean_quant(self.weight)
        return F.linear(x, w_q, self.bias)

    def ternary_loss(self):
        w_q = self.absmean_quant(self.weight)
        return F.mse_loss(self.weight, w_q)

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
        self.pos_embed = nn.Embedding(2048, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_embed.weight

    def forward(self, input_ids):
        # Cast to long to avoid embedding errors
        input_ids = input_ids.long()
        b, l = input_ids.shape
        pos = torch.arange(l, device=input_ids.device).long().unsqueeze(0).expand(b, l)
        
        x = self.tok_embed(input_ids) + self.pos_embed(pos)
        
        mask = torch.triu(torch.full((l, l), float("-inf"), device=input_ids.device), diagonal=1)
        for blk in self.blocks:
            x = blk(x, attn_mask=mask)
        return self.lm_head(self.ln_f(x))

    def ternary_loss(self):
        loss = 0
        for m in self.modules():
            if isinstance(m, BitLinear158):
                loss += m.ternary_loss()
        return loss

def main():
    print(f"🚀 Training 300M on {device}...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").select(range(20000))
    tokenized = ds.map(lambda x: tokenizer(x["text"], truncation=True, max_length=512), batched=True, remove_columns=ds.column_names)
    loader = DataLoader(tokenized, batch_size=4, shuffle=True, collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False))

    model = BitNet300M().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scaler = GradScaler('cuda')

    for epoch in range(2):
        model.train()
        total_ce, total_tern, steps = 0, 0, 0
        
        for batch in loader:
            input_ids = batch["input_ids"].to(device).long()  # Force long here
            
            with autocast('cuda', enabled=True):
                logits = model(input_ids)
                # Labels must be long too
                ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), input_ids.view(-1), ignore_index=-100)
                t_loss = model.ternary_loss()
                loss = ce_loss + 0.1 * t_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_ce += ce_loss.item()
            total_tern += t_loss.item()
            steps += 1
            
            if steps % 20 == 0:
                print(f"Epoch {epoch} Step {steps}: CE={ce_loss.item():.3f} Tern={t_loss.item():.3f}")
            if steps >= 200: break

        print(f"✅ Epoch {epoch} Avg CE: {total_ce/steps:.3f}")
        torch.save(model.state_dict(), f"bitnet_300m_epoch{epoch}.pt")

    model.eval()
    tern_pct = []
    for m in model.modules():
        if isinstance(m, BitLinear158):
            w = m.weight
            w_q = m.absmean_quant(w)
            pct = 100 * torch.isclose(w, w_q, atol=0.01).float().mean().item()
            tern_pct.append(pct)
    print(f"📊 Final Ternary Clustering: {sum(tern_pct)/len(tern_pct):.1f}%")

if __name__ == "__main__":
    main()
