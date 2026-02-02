#!/usr/bin/env python3
import os
os.environ["TORCH_CUDA_MEMPOOL_DISABLE_RESERVOIR"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from transformers import GPT2Tokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
from torch.amp import autocast, GradScaler

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model Architecture (Same as before) ---
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
        self.pos_embed = nn.Embedding(1024, d_model) # Reduced seq len for faster training
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_embed.weight

    def forward(self, input_ids):
        input_ids = input_ids.long()
        b, l = input_ids.shape
        # Truncate if longer than pos_embed
        if l > 1024: 
            input_ids = input_ids[:, :1024]
            l = 1024
            
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

# --- Streaming Dataset Wrapper ---
class FineWebStream(IterableDataset):
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Streaming load - no download wait
        self.ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)

    def __iter__(self):
        for sample in self.ds:
            # Tokenize on the fly
            tokens = self.tokenizer(sample["text"], truncation=True, max_length=self.max_length, return_tensors="pt")
            yield {"input_ids": tokens["input_ids"][0]}

def main():
    print(f"🚀 Training 300M on {device} with FineWeb-Edu (Streaming)...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Setup Streaming Data
    dataset = FineWebStream(tokenizer, max_length=512)
    loader = DataLoader(dataset, batch_size=8, collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False))

    model = BitNet300M().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4) # Slightly higher LR for fresh start
    scaler = GradScaler('cuda')

    model.train()
    total_ce, total_tern = 0, 0
    
    # Infinite loop over stream, we break manually
    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device).long()
        
        with autocast('cuda', enabled=True):
            logits = model(input_ids)
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
        
        # Logging
        if step % 20 == 0:
            print(f"Step {step}: CE={ce_loss.item():.3f} Tern={t_loss.item():.3f}")

        # Save checkpoints
        if step > 0 and step % 500 == 0:
            print(f"💾 Saving checkpoint at step {step}...")
            torch.save(model.state_dict(), f"bitnet_300m_fineweb_{step}.pt")
            
            # Quick Ternary Check
            tern_pct = []
            for m in model.modules():
                if isinstance(m, BitLinear158):
                    w = m.weight
                    w_q = m.absmean_quant(w)
                    pct = 100 * torch.isclose(w, w_q, atol=0.01).float().mean().item()
                    tern_pct.append(pct)
            print(f"📊 Ternary Clustering: {sum(tern_pct)/len(tern_pct):.1f}%")

        # Stop condition (Adjust based on your time - 5000 is min for results)
        if step >= 5000:
            print("✅ Reached 5000 steps. Training Complete.")
            torch.save(model.state_dict(), "bitnet_300m_fineweb_final.pt")
            break

if __name__ == "__main__":
    main()
