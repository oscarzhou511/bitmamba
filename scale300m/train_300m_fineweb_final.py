#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from datasets import load_dataset
from torch.amp import autocast, GradScaler
import math

os.environ["TORCH_CUDA_MEMPOOL_DISABLE_RESERVOIR"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- CONFIG ---
TOTAL_STEPS = 10000 # Run longer for better results
BATCH_SIZE = 12     # Adjust based on VRAM
LR = 4e-4

# --- MODEL (BitNet 300M) ---
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
    
    def ternary_loss(self):
        loss = 0
        for m in self.modules():
            if isinstance(m, BitLinear158): loss += m.ternary_loss()
        return loss

# --- DATA LOADER (FineWeb Streaming) ---
def get_fineweb_loader(tokenizer, batch_size=8):
    print("📚 Initializing FineWeb-Edu Stream...")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    
    # 1. Skip first 50k to avoid overfit buffer from previous runs
    ds = ds.skip(50000)
    # 2. Shuffle buffer to ensure randomness
    ds = ds.shuffle(buffer_size=10000, seed=42)
    
    iterator = iter(ds)
    
    while True:
        batch_inputs = []
        try:
            for _ in range(batch_size):
                item = next(iterator)
                text = item['text']
                if len(text) < 200: continue
                
                # Random crop to avoid header bias
                if len(text) > 2000:
                    start = torch.randint(0, len(text)-1500, (1,)).item()
                    text = text[start:start+1500]
                
                tokens = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
                batch_inputs.append(tokens["input_ids"][0])
                
        except StopIteration:
            break
            
        if not batch_inputs: continue
        
        padded = torch.nn.utils.rnn.pad_sequence(batch_inputs, batch_first=True, padding_value=tokenizer.eos_token_id)
        yield padded

# --- TRAINING LOOP ---
def main():
    print(f"🚀 Training 300M on FineWeb-Edu (Production Run)...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    model = BitNet300M().to(device)
    
    # Init Weights
    def init_weights(m):
        if isinstance(m, nn.Linear): torch.nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding): torch.nn.init.normal_(m.weight, std=0.02)
    model.apply(init_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS, eta_min=1e-5)
    scaler = GradScaler('cuda')
    
    train_loader = get_fineweb_loader(tokenizer, batch_size=BATCH_SIZE)
    
    model.train()
    
    for step, input_ids in enumerate(train_loader):
        if step >= TOTAL_STEPS: break
        
        input_ids = input_ids.to(device).long()
        
        with autocast('cuda', enabled=True):
            logits = model(input_ids)
            
            # --- THE FIX (Shifted Labels) ---
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1), 
                ignore_index=tokenizer.eos_token_id
            )
            
            t_loss = model.ternary_loss()
            loss = ce_loss + 0.05 * t_loss 

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % 20 == 0:
            print(f"Step {step}: CE={ce_loss.item():.3f} Tern={t_loss.item():.3f}")

        if step > 0 and step % 2000 == 0:
            print(f"💾 Saving Checkpoint {step}...")
            torch.save(model.state_dict(), f"bitnet_300m_fineweb_final_{step}.pt")

    torch.save(model.state_dict(), "bitnet_300m_fineweb_final.pt")
    print("✅ Training Complete.")

if __name__ == "__main__":
    main()
