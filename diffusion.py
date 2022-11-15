import torch
from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"


pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision="fp16")
pipe = pipe.to(device)

with open('prompt.txt', 'r') as f:
    prompts = f.read().splitlines()   
    for prompt in prompts:
        image = pipe(prompt).images[0]  
        image.save("img/" + prompt.replace(" ", "_") + "png")
