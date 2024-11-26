from OmniGen import OmniGenPipeline
import os, gradio

model_group = 'shitao'
model = ''
MODEL_PATH = r'E:\04.Code\HuggingFace\models'

def test_om():
    model_id = os.path.join(MODEL_PATH, model_group,  'OmniGen-v1')
    print(model_id)
    pipe = OmniGenPipeline.from_pretrained(model_id)

    # Text to Image
    print("pipe .....")
    images = pipe(
        prompt="A curly-haired man in a red shirt is drinking tea.", 
        height=1024, 
        width=1024, 
        guidance_scale=2.5,
        seed=0,
    )
    print('save ....')
    images[0].save("example_t2i.png")  # save output PIL Image

import gradio as gr
import torch
from PIL import Image
import numpy as np



def image_to_image(
    init_image,
    prompt,
    negative_prompt="",
):
    print("1,", init_image, prompt, negative_prompt)
    if init_image is None:
        return None
    print("2", init_image, prompt, negative_prompt)
    # 确保图像是PIL格式
    if isinstance(init_image, np.ndarray):
        init_image = Image.fromarray(init_image)
    
    print("3", init_image, prompt, negative_prompt)
    
 

# 创建Gradio界面
with gr.Blocks() as demo:
    gr.Markdown("## Stable Diffusion 图生图Demo")
    
    with gr.Row():
        with gr.Column():
            # 输入区域
            input_image = gr.Image(label="上传初始图片", type="filepath")
            prompt = gr.Textbox(label="正面提示词", placeholder="请输入正面提示词...")
            negative_prompt = gr.Textbox(label="负面提示词", placeholder="请输入负面提示词...")
            
            generate_btn = gr.Button("生成图片")
        
        with gr.Column():
            # 输出区域
            output_image = gr.Image(label="生成结果")
            output_seed = gr.Number(label="使用的种子值", interactive=False)
    
    # 设置点击事件
    generate_btn.click(
        fn=image_to_image,
        inputs=[
            input_image,
            prompt,
            negative_prompt,
        ],
        outputs=[output_image, output_seed]
    )
    

# 启动界面
demo.launch()