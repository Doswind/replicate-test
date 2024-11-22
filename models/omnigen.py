import os
import gradio as gr

from OmniGen import OmniGenPipeline

from .config import MODEL_PATH, APP_ROOT

model_group = "shitao"

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

def process_image(model, prompt, img1, img2, img3, parameters):

    model_id = os.path.join(MODEL_PATH, model_group,  model)
    print(type(img1))
    print(img1)
    return 

    pipe = OmniGenPipeline.from_pretrained(model_id)

    # Text to Image
    images = pipe(
        prompt="A curly-haired man in a red shirt is drinking tea.", 
        height=1024, 
        width=1024, 
        guidance_scale=2.5,
        seed=0,
    )
    images[0].save("example_t2i.png")  # save output PIL Image

    # Multi-modal to Image
    # In prompt, we use the placeholder to represent the image. The image placeholder should be in the format of <img><|image_*|></img>
    # You can add multiple images in the input_images. Please ensure that each image has its placeholder. For example, for the list input_images [img1_path, img2_path], the prompt needs to have two placeholders: <img><|image_1|></img>, <img><|image_2|></img>.
    images = pipe(
        prompt="A man in a black shirt is reading a book. The man is the right man in <img><|image_1|></img>.",
        input_images=["./imgs/test_cases/two_man.jpg"],
        height=1024, 
        width=1024,
        separate_cfg_infer=False,  # if OOM, you can set separate_cfg_infer=True 
        guidance_scale=3, 
        img_guidance_scale=1.6
    )

    return images


def run(model, prompt, img1, img2, img3,
        height, width, guidance_scale,
        img_guidance_scale, inference_steps,
        seed, randomize,
        max_input_image_size,
        offload_model, 
        use_input_image_size):
    
    parameters = {"width": int(width), 
                  "height": int(height), 
                  "seed": int(seed), 
                  "randomize": bool(randomize),
                  "offload_model": offload_model,
                  "use_input_image_size": use_input_image_size,
                  "img_guidance_scale": float(img_guidance_scale), 
                  "max_input_image_size": int(max_input_image_size)}   

    return process_image(model, prompt, img1, img2, img3, parameters)
                      

def interface():
    with gr.Blocks(css=display_css) as demo:
        with gr.Column():
            prompt = gr.Textbox(
                label="Enter your prompt, use <img><|image_i|></img> to represent i-th input image",
                placeholder="Type your prompt here...",
                lines=2
            )
            
            with gr.Row(elem_classes="image-row"):
                with gr.Column(elem_classes="image-container"):
                    img1 = gr.Image(
                        label="<img><|image_1|></img>",
                        type="filepath",
                        #elem_classes="upload-button"
                    )
                    print("herereree")
                    print(img1)
                    print(type(img1))
                with gr.Column(elem_classes="image-container"):
                    img2 = gr.Image(
                        label="<img><|image_2|></img>",
                        type="filepath",
                        #elem_classes="upload-button"
                    )
                with gr.Column(elem_classes="image-container"):
                    img3 = gr.Image(
                        label="<img><|image_3|></img>",
                        type="filepath",
                        #elem_classes="upload-button"
                    )
            
            with gr.Accordion("Advanced Settings", open=False, elem_classes="advanced-options"):
                height = gr.Slider(
                    minimum=128,
                    maximum=2048,
                    value=1024,
                    step=16,
                    label="Height"
                )
                
                width = gr.Slider(
                    minimum=128,
                    maximum=2048,
                    value=1024,
                    step=16,
                    label="Width"
                )
                
                guidance_scale = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=2.5,
                    step=0.1,
                    label="Guidance Scale"
                )
                
                img_guidance_scale = gr.Slider(
                    minimum=1,
                    maximum=2,
                    value=1.9,
                    step=0.1,
                    label="img_guidance_scale"
                )
                
                inference_steps = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=50,
                    step=1,
                    label="Inference Steps"
                )
 
                seed = gr.Slider(
                    value=42,
                    label="Seed",
                    minimum=0,
                    maximum=2147483647    
                )
                randomize = gr.Checkbox(label="Randomize seed", value=True)
                
                max_input_image_size = gr.Slider(
                    value=1024,
                    label="max_input_image_size",
                    minimum=128,
                    maximum=2048  
                )

                #with gr.Column(elem_classes="checkbox-group"):
                    
                gr.Markdown(
                    "Whether to use separate inference process for different guidance. This will reduce the memory cost.",
                    elem_classes="markdown-text"
                )
                separate_cfg_infer = gr.Checkbox(
                    label="separate_cfg_infer",
                    value=True,
                    #elem_classes="checkbox-item"
                )

                gr.Markdown(
                    "Offload model to CPU, which will significantly reduce the memory cost but slow down the generation speed. You can cancel separate_cfg_infer and set offload_model=True. If both separate_cfg_infer and offload_model are True, further reduce the memory, but slowest generation",
                    elem_classes="markdown-text"
                )

                offload_model = gr.Checkbox(
                    label="offload_model",
                    value=False,
                    #elem_classes="checkbox-item"
                )
                
                gr.Markdown(
                    "Automatically adjust the output image size to be same as input image size. For editing and controlnet task, it can make sure the output image has the same size as input image leading to better performance",
                    elem_classes="markdown-text"
                )
                use_input_image_size = gr.Checkbox(
                    label="use_input_image_size_as_output",
                    value=False,
                    #elem_classes="checkbox-item"
                )
                    
            
            generate_btn = gr.Button("Generate Image", elem_classes="submit-button")
            
            output_image = gr.Image(
                label="Output Image",
                interactive=False,
                show_label=False
            )

            # 添加点击事件处理
            generate_btn.click(
                fn=run,
                inputs=[
                    img1, img2, img3,
                    height, width, guidance_scale,
                    img_guidance_scale, inference_steps,
                    seed, randomize,
                    max_input_image_size,
                    offload_model, 
                    use_input_image_size
                ],
                outputs=[output_image]
            )
            
            return demo

# 启动接口
if __name__ == "__main__":
    demo = interface()
    demo.launch()
