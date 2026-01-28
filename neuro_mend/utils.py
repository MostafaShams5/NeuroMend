import torch
import gc
import os
from PIL import Image

def flush_vram():
    """Aggressively clears CUDA cache and garbage collects."""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()
    print("[-] VRAM Flushed.")

def load_image_batch(folder_path, valid_extensions=(".jpg", ".png", ".jpeg")):
    """Helper to get image paths."""
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
            if f.lower().endswith(valid_extensions)]
