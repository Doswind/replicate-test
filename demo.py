import torch
from diffusers import StableDiffusion3Pipeline
from diffusers import StableDiffusion3Img2ImgPipeline
from diffusers.utils import load_image

# refer:https://www.cnblogs.com/huggingface/p/18253377
# model_path = 'E:\04.Code\HuggingFace\models\hub\models--stabilityai--stable-diffusion-3-medium-diffusers'
model_path = "D:\\huggingface\\models\\stabilityai\\stable-diffusion-3-medium-diffusers"
#model_path = "D:\\huggingface\\models\\stabilityai\\stable-diffusion-3.5-large-turbo"

def text2image():
    pipe = StableDiffusion3Pipeline.from_pretrained(model_path, local_files_only=True, text_encoder_3=None, tokenizer_3=None,  torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    
    prompt = "A capybara holding a sign that reads Hello World"
    #prompt = "Ultraman in action, dynamic battle scene, fighting giant kaiju monster, cosmic superhero, silver armored suit, glowing color timer chest light, heroic pose, energy beam attack, urban destruction background, dramatic lighting, epic scale, motion blur, detailed special effects, cinematic composition, high-quality render, dynamic angle, dramatic perspective, sparks and explosions, detailed textures, 8k resolution, hyper-detailed, masterpiece"
    image = pipe(
        prompt,
        num_inference_steps=40,
        guidance_scale=4.5,
    ).images[0]
    image.save("capybara.png")

def image2image():
    pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.float16)
    #pipe = StableDiffusion3Img2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    
    #init_image = load_image("https://cn.bing.com/images/search?view=detailV2&ccid=8UdRWhtX&id=A597B09CF3A3F4BE44327EA8871FD665AD7DE3BA&thid=OIP.8UdRWhtXzEB3cDcffOJWNwHaHa&mediaurl=https%3a%2f%2fc-ssl.duitang.com%2fuploads%2fblog%2f202208%2f01%2f20220801163120_35a33.jpeg&exph=1115&expw=1115&q=%e5%a5%b3%e5%ad%a9%e5%a4%b4%e5%83%8f&simid=608052217692563920&FORM=IRPRST&ck=5FC433E94C68AB362BC3EFBA462C2806&selectedIndex=3&itb=0")
    init_image = load_image("C:\\Users\\Administrator\\Pictures\\OIP-C.jpg")
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

image2image()
#text2image()
