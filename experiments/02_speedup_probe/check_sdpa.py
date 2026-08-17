import torch
print("torch", torch.__version__)
print("flash_sdp", torch.backends.cuda.flash_sdp_enabled())
print("mem_efficient_sdp", torch.backends.cuda.mem_efficient_sdp_enabled())
print("math_sdp", torch.backends.cuda.math_sdp_enabled())
print("cap", torch.cuda.get_device_capability())
print("flash_attn_import", end=" ")
try:
    import flash_attn  # noqa
    print("yes", flash_attn.__version__)
except Exception as e:
    print("no", type(e).__name__)
