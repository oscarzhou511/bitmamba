import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from train_300m_bitnet import BitNet300M, BitLinear158, BitFFN, TransformerBlock

# Re-define model class structure if not importable, 
# but since you're in the same dir, import should work if code is saved.
# If not, paste the classes here too.

model = BitNet300M().cuda()
# Load your epoch 1 checkpoint
model.load_state_dict(torch.load("bitnet_300m_epoch1.pt"), strict=False)
model.eval()

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def generate(prompt, max_new=30):
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].cuda().long()
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(input_ids)
            next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

print("🤖 300M BitNet (41% Ternary):")
print(generate("The capital of France is"))
print(generate("Artificial intelligence is"))
