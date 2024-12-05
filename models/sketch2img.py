from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL
from controlnet_aux.pidi import PidiNetDetector
import torch
from PIL import Image
import numpy as np


adapter = T2IAdapter.from_pretrained(
    "TencentARC/t2i-adapter-sketch-sdxl-1.0", torch_dtype=torch.float16, variant="fp16"
).to("cuda")

euler_a = EulerAncestralDiscreteScheduler.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler"
)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    adapter=adapter,
    scheduler=euler_a,
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

pipe.enable_xformers_memory_efficient_attention()
pidinet = PidiNetDetector.from_pretrained("lllyasviel/Annotators").to("cuda")

def preprocess_sketch(image):
    image_np = np.array(image.convert("RGB"))
    return pidinet(image_np, detect_resolution=1024, image_resolution=1024, apply_filter=True)

def generate_from_prompt(sketch_image, prompt):
    processed_sketch = preprocess_sketch(sketch_image)
    generated_image = pipe(prompt=prompt, image=processed_sketch, num_inference_steps=50, adapter_conditioning_scale=0.9,
    guidance_scale=7.5, negative_prompt = "extra digit, fewer digits, cropped, worst quality, low quality, glitch, deformed, mutated, ugly, disfigured"
).images[0]
    return generated_image

def get_random_sample():
    random_drawing = Image.new('RGB', (256, 256), 'white')
    return random_drawing
