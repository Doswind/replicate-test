import gradio as gr

def generate_image(text_input,
            seed,
            width,
            height,
            guidance_scale,
            num_steps):
    # 处理逻辑在这里实现
    return "processed_image"

def interface():
    with gr.Column():
        # 文本输入和运行按钮行
        with gr.Row():
            text_input = gr.Textbox(
                label="", 
                placeholder="Enter your prompt",
                show_label=False,
            )
            run_btn = gr.Button("Run", variant="secondary")
        
        # 图像输出区域
        image_output = gr.Image(height=400)
        
        # 高级设置折叠面板
        with gr.Accordion("Advanced Settings"):
            # Seed设置
            with gr.Group():
                with gr.Row():
                    seed = gr.Number(label="Seed", value=0)
                    seed_btn = gr.Button("🔄", elem_classes="tool-button")
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                seed_slider = gr.Slider(minimum=0, maximum=2147483647, step=1, label="", show_label=False)
            
            # 宽度和高度设置
            with gr.Group():
                with gr.Row():
                    with gr.Column():
                        width = gr.Number(label="Width", value=1024)
                        width_slider = gr.Slider(minimum=256, maximum=2048, value=1024, step=1, label="", show_label=False)
                    with gr.Column():
                        height = gr.Number(label="Height", value=1024)
                        height_slider = gr.Slider(minimum=256, maximum=2048, value=1024, step=1, label="", show_label=False)
            
            # Guidance Scale和推理步数设置
            with gr.Group():
                with gr.Row():
                    with gr.Column():
                        guidance_scale = gr.Number(label="Guidance Scale", value=3.5)
                        guidance_slider = gr.Slider(minimum=1, maximum=15, value=3.5, step=0.1, label="", show_label=False)
                    with gr.Column():
                        num_steps = gr.Number(label="Number of inference steps", value=28)
                        steps_slider = gr.Slider(minimum=1, maximum=50, value=28, step=1, label="", show_label=False)
        
        # 示例提示词
        gr.Examples(
            examples=[
                "a tiny astronaut hatching from an egg on the moon",
                "a cat holding a sign that says hello world",
                "an anime illustration of a wiener schnitzel"
            ],
            inputs=text_input,
        )

    # 组件交互逻辑
    def update_number_input(slider_value):
        return gr.Number.update(value=slider_value)
    
    # 绑定slider和number input的双向更新
    width_slider.change(fn=update_number_input, inputs=[width_slider], outputs=[width])
    width.change(fn=lambda x: gr.Slider.update(value=x), inputs=[width], outputs=[width_slider])
    
    height_slider.change(fn=update_number_input, inputs=[height_slider], outputs=[height])
    height.change(fn=lambda x: gr.Slider.update(value=x), inputs=[height], outputs=[height_slider])
    
    guidance_slider.change(fn=update_number_input, inputs=[guidance_slider], outputs=[guidance_scale])
    guidance_scale.change(fn=lambda x: gr.Slider.update(value=x), inputs=[guidance_scale], outputs=[guidance_slider])
    
    steps_slider.change(fn=update_number_input, inputs=[steps_slider], outputs=[num_steps])
    num_steps.change(fn=lambda x: gr.Slider.update(value=x), inputs=[num_steps], outputs=[steps_slider])

    # 运行按钮点击事件
    run_btn.click(
        fn=generate_image,
        inputs=[
            text_input,
            seed,
            width,
            height,
            guidance_scale,
            num_steps
        ],
        outputs=image_output
    ) 