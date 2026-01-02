import torch
import sys

print("--- SISTEM CHECK ---")
print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print("✅ STATUS: AMAN. GPU SIAP DIGUNAKAN UNTUK SKRIPSI!")
else:
    print("❌ STATUS: BAHAYA. GPU TIDAK TERBACA PYTHON.")