import json

c = json.load(open("/root/cr-nn/data/qwen0.5b/config.json"))
L = c.get("num_hidden_layers")
n_heads = c.get("num_attention_heads")
n_kv = c.get("num_key_value_heads")
hd = c.get("head_dim", c.get("hidden_size", 0) // n_heads)
hidden = c.get("hidden_size")
print(f"Qwen2.5-0.5B: layers={L} heads={n_heads} kv_heads={n_kv} "
      f"head_dim={hd} hidden={hidden}")

# KV cache: 2 (K+V) x L x n_kv x head_dim x 2 bytes (bf16)
bytes_per_token = 2 * L * n_kv * hd * 2
print(f"KV cache per token = {bytes_per_token} bytes = {bytes_per_token/1024:.1f} KB")

for budget_gb in (8.0, 40.0, 80.0):
    max_tokens = budget_gb * 1e9 / bytes_per_token
    print(f"  at {budget_gb:>4.0f}GB: max {max_tokens/1e3:.0f}K tokens")

n = 1_000_000
print(f"  at 1M tokens: {bytes_per_token * n / 1e9:.1f} GB")

# CR block-recurrent O(1) state for contrast
print("CR block-recurrent running state: O(1) = 0.015 GB at 1.36M tokens "
      "(from block_recurrent.py)")
