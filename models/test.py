import gradio as gr


def process_image(image, height, width, guidance_scale, img_guidance_scale, inference_steps, seed, use_random_seed, 
                 separate_cfg_infer, offload_model, use_input_image_size):
    # 处理逻辑在这里实现
    return "processed_image"

with gr.Blocks() as demo:
    with gr.Column():
        prompt = gr.Textbox(
            label="",
            placeholder="Enter your prompt, use <img><|image_i|></img> to represent i-th input image",
            lines=2
        )
        
        with gr.Row(elem_classes="image-row"):
            with gr.Column(elem_classes="image-container"):
                img1 = gr.Image(
                    label="<img><|image_1|></img>",
                    type="filepath",
                    elem_classes="upload-button"
                )
            with gr.Column(elem_classes="image-container"):
                img2 = gr.Image(
                    label="<img><|image_2|></img>",
                    type="filepath",
                    elem_classes="upload-button"
                )
            with gr.Column(elem_classes="image-container"):
                img3 = gr.Image(
                    label="<img><|image_3|></img>",
                    type="filepath",
                    elem_classes="upload-button"
                )
        
        with gr.Accordion("Advanced Options", open=False, elem_classes="advanced-options"):
            height = gr.Slider(
                minimum=128,
                maximum=2048,
                value=1024,
                step=1,
                label="Height"
            )
            
            width = gr.Slider(
                minimum=128,
                maximum=2048,
                value=1024,
                step=1,
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
            
            with gr.Row():
                seed = gr.Number(
                    value=128,
                    label="Seed",
                    minimum=0,
                    maximum=2147483647
                )
                randomize = gr.Checkbox(label="Randomize seed")
            
            with gr.Column(elem_classes="checkbox-group"):
                separate_cfg_infer = gr.Checkbox(
                    label="separate_cfg_infer",
                    value=True,
                    elem_classes="checkbox-item"
                )
                gr.Markdown(
                    "Whether to use separate inference process for different guidance. This will reduce the memory cost.",
                    elem_classes="checkbox-label"
                )
                
                offload_model = gr.Checkbox(
                    label="offload_model",
                    value=False,
                    elem_classes="checkbox-item"
                )
                gr.Markdown(
                    "Offload model to CPU, which will significantly reduce the memory cost but slow down the generation speed. You can cancel separate_cfg_infer and set offload_model=True. If both separate_cfg_infer and offload_model are True, further reduce the memory, but slowest generation",
                    elem_classes="checkbox-label"
                )
                
                use_input_image_size = gr.Checkbox(
                    label="use_input_image_size_as_output",
                    value=False,
                    elem_classes="checkbox-item"
                )
                gr.Markdown(
                    "Automatically adjust the output image size to be same as input image size. For editing and controlnet task, it can make sure the output image has the same size as input image leading to better performance",
                    elem_classes="checkbox-label"
                )
        
        generate_btn = gr.Button("Generate Image", elem_classes="generate-button")
        
        output_image = gr.Image(
            label="Output Image",
            elem_classes="output-image",
            interactive=False,
            show_label=False
        )

demo.launch()