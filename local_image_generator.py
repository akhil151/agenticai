import os
import logging
import torch
from diffusers import AutoPipelineForText2Image
from datetime import datetime

logger = logging.getLogger(__name__)

# Global variable to hold the pipeline so we don't reload it
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "❌ CUDA is not available! You have the CPU-only version of PyTorch installed. "
                "Please run: pip uninstall -y torch torchvision torchaudio\n"
                "Then run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
            )
            
        logger.info("⏳ Loading Stable Diffusion 1.5 locally from default cache...")
        _pipeline = AutoPipelineForText2Image.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        # Optimizations for 4GB VRAM
        logger.info("⚙️ Applying memory optimizations for 4GB VRAM...")
        _pipeline.enable_model_cpu_offload()
        _pipeline.enable_attention_slicing()
    return _pipeline

def generate_local_image(prompt: str, cache_dir: str, fallback_func) -> str:
    """Generate image locally and return file path."""
    try:
        pipe = get_pipeline()
        logger.info(f"🎨 Generating image for prompt: '{prompt[:40]}...' (This will take time on 4GB VRAM)")
        
        # Num inference steps reduced slightly to speed it up
        image = pipe(prompt=prompt, num_inference_steps=25).images[0]
        
        safe = "".join(c for c in prompt if c.isalnum() or c in (" ", "-", "_")).rstrip()
        safe = safe.replace(" ", "_")[:50]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(cache_dir, f"{safe}_{ts}.png")
        image.save(path)
        logger.info(f"✅ Local image saved: {path}")
        return path
    except Exception as e:
        logger.error(f"❌ Local image generation exception: {e}")
        return fallback_func()