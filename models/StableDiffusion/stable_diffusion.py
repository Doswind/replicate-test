import os
import torch
from diffusers import StableDiffusion3Pipeline, DPMSolverMultistepScheduler
from diffusers import StableDiffusion3Img2ImgPipeline
from diffusers.utils import load_image

from .config import MODEL_PATH, APP_ROOT

import gradio as gr

model_group = "BAAI"
#model_path = "D:\\huggingface\\models\\stabilityai"

display_css = """
.submit-button {
    background: linear-gradient(90deg, rgba(255,165,0,1) 0%, rgba(255,200,100,1) 100%);
    color: #D35400;
    border: none;
    padding: 10px 20px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
    border-radius: 12px;
    font-weight: bold;
}
.markdown-text {
    color: #666666;
    font-size: 0.9em;
}
"""

def text2image(model, prompt, parameters):
    model_id = os.path.join(MODEL_PATH, model_group,  model)
    print(model_id)
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


def run(model, prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps):

    parameters = {"width": int(width), 
                  "height": int(height), 
                  "seed": int(seed), 
                  "randomize_seed": bool(randomize_seed),
                  "randomize_seed": int(randomize_seed), 
                  "guidance_scale": float(guidance_scale), 
                  "num_inference_steps": int(num_inference_steps)}   

    return text2image(model, prompt, parameters)


def interface():
    with gr.Blocks(css=display_css) as demo:
        with gr.Column():
            model = gr.Dropdown(
                label="Select your model", 
                choices=["stable-diffusion-3.5-medium-diffusers", "stable-diffusion-3.5-large-turbo"],
                value="stable-diffusion-3.5-medium-diffusers"
            )
            prompt = gr.Textbox(label="Enter your prompt", placeholder="目前对英文的支持较好，请使用英文")
            output_image_sd = gr.Image(label="Output")
            run_button = gr.Button(value="Run", elem_classes="submit-button")
            with gr.Accordion("Advanced Settings", open=False):
                seed = gr.Slider(label="Seed", minimum=0, maximum=1000000, step=1, value=0)
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                with gr.Row():
                    with gr.Column():
                        width = gr.Slider(label="Width", minimum=512, maximum=1440, step=16, value=1024)
                    with gr.Column():
                        height = gr.Slider(label="Height", minimum=512, maximum=1440, step=16, value=1024)
                with gr.Row():
                    with gr.Column():
                        guidance_scale = gr.Slider(label="Guidance scale", minimum=0, maximum=7.5, step=0.1, value=4.5)
                    with gr.Column():
                        num_inference_steps = gr.Slider(label="Number of inference steps", minimum=1, maximum=50, step=1, value=40)
            gr.Examples(examples=["A beautiful sunset over a mountain range", "A futuristic cityscape at night"], inputs=prompt) 

            run_button.click(run, inputs=[model, prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps], outputs=output_image_sd)

    
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

    demo = interface()
    demo.launch()