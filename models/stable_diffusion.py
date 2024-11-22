import os
import torch
from diffusers import StableDiffusion3Pipeline, DPMSolverMultistepScheduler
from diffusers import StableDiffusion3Img2ImgPipeline
from diffusers.utils import load_image

from .config import MODEL_PATH, APP_ROOT

model_group = "stabilityai"
#model_path = "D:\\huggingface\\models\\stabilityai"

def text2image(model, prompt, parameters):
    model_id = os.path.join(MODEL_PATH, model_group,  model)

    pipe = StableDiffusion3Pipeline.from_pretrained(model_id, local_files_only=True, text_encoder_3=None, tokenizer_3=None,  torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    
    #prompt = "A capybara holding a sign that reads Hello World"
    #prompt = "Ultraman in action, dynamic battle scene, fighting giant kaiju monster, cosmic superhero, silver armored suit, glowing color timer chest light, heroic pose, energy beam attack, urban destruction background, dramatic lighting, epic scale, motion blur, detailed special effects, cinematic composition, high-quality render, dynamic angle, dramatic perspective, sparks and explosions, detailed textures, 8k resolution, hyper-detailed, masterpiece"
    image = pipe(
        prompt,
        width=parameters['width'],
        height=parameters['height'],
        seed=parameters['seed'],
        randomize_seed=parameters['randomize_seed'],
        guidance_scale=parameters['guidance_scale'],
        num_inference_steps=parameters['num_inference_steps']
    ).images[0]

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



if __name__ == '__main__':
    model = "stabilityai/stable-diffusion-3-medium-diffusers"
    prompt = ""
    parameters = {
        "width": 1024,
        "height": 1024,
        "guidance_scale": 5,
        "num_inference_steps": 28,
        "seed": 0 ,
        "randomize_seed": 0
    }

    text2image(model, prompt, parameters)