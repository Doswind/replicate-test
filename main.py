import gradio as gr
from PIL import Image 

#from models import stable_diffusion as sd
from models import stable_diffusion, omnigen, flux

custom_css = """
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

.clear-button {
    background-color: #ECECEC;
    color: #333;
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

"""

def process_codeformer(image, pre_face_align, background_enhance, face_upsample, rescaling_factor, codeformer_fidelity):
    # 这里放置 Codeformer 的图像处理逻辑
    # 目前只是返回输入图像作为示例
    return image

def process_gfpgan(image, version, rescaling_factor):
    # 这里放置 GFPGAN 的图像处理逻辑
    # 目前只是返回输入图像作为示例
    return image, image, image


def generate_image():
    pass

# 文本生成相关函数
def qwen_generate(prompt):
    return f"QWen response: {prompt}"

def llama_generate(prompt):
    return f"Llama response: {prompt}"

# 音频生成相关函数
def tts_generate(text):
    # 示例返回音频
    return None

# 视频生成相关函数
def video_generate(prompt):
    # 示例返回视频
    return None


def table_text_generate():
    with gr.Tabs():
        with gr.Tab("QWen"):
            text_input1 = gr.Textbox(label="输入提示词")
            text_button1 = gr.Button("生成")
            text_output1 = gr.Textbox(label="生成结果")
            text_button1.click(qwen_generate, inputs=text_input1, outputs=text_output1)
        
        with gr.Tab("Llama"):
            text_input2 = gr.Textbox(label="输入提示词")
            text_button2 = gr.Button("生成")
            text_output2 = gr.Textbox(label="生成结果")
            text_button2.click(llama_generate, inputs=text_input2, outputs=text_output2)

def table_image_generate():
    with gr.Tabs():
        with gr.TabItem("Stable Diffusion"):
            stable_diffusion.interface()
             
        with gr.TabItem("OmniGen-V1"):
            print('OmniGen-V1  .......')
            omnigen.interface()
        
        with gr.TabItem("FLUX.1-dev"):
            flux.interface()   
            
        with gr.TabItem("Codeformer"):
            with gr.Row():
                with gr.Column():
                    input_image_cf = gr.Image(label="Input", type="numpy", interactive=True)
                    pre_face_align_cf = gr.Checkbox(label="Pre_Face_Align", value=True)
                    background_enhance_cf = gr.Checkbox(label="Background_Enhance", value=True)
                    face_upsample_cf = gr.Checkbox(label="Face_Upsample", value=True)
                    rescaling_factor_cf = gr.Number(label="Rescaling_Factor (up to 4)", value=2, precision=0)
                    codeformer_fidelity_cf = gr.Slider(label="Codeformer_Fidelity (0 for better quality, 1 for better identity)", minimum=0, maximum=1, value=0.5, step=0.01)
                    with gr.Row():
                        clear_button_cf = gr.Button(value="Clear", elem_classes="clear-button")
                        submit_button_cf = gr.Button(value="Submit", elem_classes="submit-button")
                with gr.Column():
                    output_image_cf = gr.Image(label="Output")

            clear_button_cf.click(lambda: [None, True, True, True, 2, 0.5, None], inputs=[], outputs=[input_image_cf, pre_face_align_cf, background_enhance_cf, face_upsample_cf, rescaling_factor_cf, codeformer_fidelity_cf, output_image_cf])
            submit_button_cf.click(process_codeformer, inputs=[input_image_cf, pre_face_align_cf, background_enhance_cf, face_upsample_cf, rescaling_factor_cf, codeformer_fidelity_cf], outputs=output_image_cf)
        
        with gr.TabItem("GFPGAN"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_image_gf = gr.Image(label="Input", type="numpy", interactive=True)
                    version_gf = gr.Radio(label="version", choices=["v1.2", "v1.3", "v1.4", "RestoreFormer"], value="v1.4")
                    rescaling_factor_gf = gr.Number(label="Rescaling factor", value=2, precision=0)
                    with gr.Row():
                        clear_button_gf = gr.Button(value="Clear", elem_classes="clear-button")
                        submit_button_gf = gr.Button(value="Submit", elem_classes="submit-button")
                with gr.Column(scale=1):
                    output_image_gf = gr.Image(label="Output (The whole image)")
                    download_output_gf = gr.File(label="Download the output image")

            clear_button_gf.click(lambda: [None, "v1.4", 2, None, None], inputs=[], outputs=[input_image_gf, version_gf, rescaling_factor_gf, output_image_gf, download_output_gf])
            submit_button_gf.click(process_gfpgan, inputs=[input_image_gf, version_gf, rescaling_factor_gf], outputs=[output_image_gf, download_output_gf, download_output_gf])
        
def table_image_analysis():
    with gr.Tabs():
        with gr.TabItem("音频生成"):
            audio_input = gr.Textbox(label="输入文本")
            audio_button = gr.Button("生成语音")
            audio_output = gr.Audio(label="生成结果")
            audio_button.click(tts_generate, inputs=audio_input, outputs=audio_output)

def table_audio_generate():
    with gr.Tabs():
        with gr.TabItem("音频生成"):
            audio_input = gr.Textbox(label="输入文本")
            audio_button = gr.Button("生成语音")
            audio_output = gr.Audio(label="生成结果")
            audio_button.click(tts_generate, inputs=audio_input, outputs=audio_output)

def table_video_generate():
    with gr.Tabs():
        with gr.TabItem("视频生成"):
            video_input = gr.Textbox(label="输入提示词")
            video_button = gr.Button("生成视频")
            video_output = gr.Video(label="生成结果")
            video_button.click(video_generate, inputs=video_input, outputs=video_output)

def table_model_training():
    pass

with gr.Blocks(css=custom_css) as demo:
    with gr.Column(scale=4):
        # 文本生成页面
        with gr.TabItem("文本生成"):
            table_text_generate()

        with gr.TabItem("图像生成"):
            table_image_generate()

        with gr.TabItem("图像理解"):
            table_image_analysis()

        with gr.TabItem("音频生成"):
            table_audio_generate()

        with gr.TabItem("视频生成"):
            table_video_generate()

        with gr.TabItem("模型训练"):
            table_model_training()

demo.launch(server_name="0.0.0.0")
