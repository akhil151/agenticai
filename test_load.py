import torch
from diffusers import AutoPipelineForText2Image
import traceback

print("Starting test...")
try:
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    print("Loaded successfully!")
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
