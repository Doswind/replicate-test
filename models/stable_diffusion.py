import os
import torch
from diffusers import StableDiffusion3Pipeline, DPMSolverMultistepScheduler
from diffusers import StableDiffusion3Img2ImgPipeline
from diffusers.utils import load_image

from models.config import MODEL_PATH, APP_ROOT


def text2image(model, prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps):
    model_id = os.path.join(MODEL_PATH, model)
    pipe = StableDiffusion3Pipeline.from_pretrained(model_id, local_files_only=True, text_encoder_3=None, tokenizer_3=None, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda") if torch.cuda.is_available() else pipe
    
    generator = torch.Generator('cuda').manual_seed(seed) if torch.cuda.is_available() else torch.Generator('cpu').manual_seed(seed)

    image = pipe(prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
        negative_prompt=negative_prompt,
        randomize_seed=randomize_seed).images[0]
    image.save('output.png')
    return image

def image2image(model):
    model_id = os.path.join(MODEL_PATH, model)
    pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(model_id, local_files_only=True, torch_dtype=torch.float16)
    #pipe = StableDiffusion3Img2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    
    init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")
    #init_image = load_image("cat.png")
    prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"
    image = pipe(prompt, image=init_image).images[0]
    image.save('new_output.png')

def list_modesl():
    from huggingface_hub import scan_cache_dir
    cache_info = scan_cache_dir()
    print(f"缓存根目录: {cache_info}")
    # 列出所有下载的模型
    for repo in cache_info.repos:
        #if "stable-diffusion-3" in repo.repo_id:
        print(f"\n模型路径: {repo.repo_path}")


def run(model, prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps):
    text2image(model, prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps)

if __name__ == '__main__':
    model = "stabilityai/stable-diffusion-3-medium-diffusers"
    width = 1024
    height = 1024
    guidance_scale = 5
    num_inference_steps = 28
    seed = 0 # or None for random seed
    prompt = ""
    negative_prompt = ""
    randomize_seed = 0

    run(model, prompt, model, prompt, negative_prompt, seed, randomize_seed, width, height, 
        guidance_scale, num_inference_steps, seed, randomize_seed, width, height, guidance_scale, num_inference_steps)