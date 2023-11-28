import gradio as gr
import numpy as np
import os
from train import *

example_inputs = [[
        "A DSLR photo of a Rugged, vintage-inspired hiking boots with a weathered leather finish, best quality, 4K, HD.",
        "Rugged, vintage-inspired hiking boots with a weathered leather finish."
    ], [
        "a DSLR photo of a Cream Cheese Donut.",
        "a Donut."
    ], [
        "A durian, 8k, HDR.",
        "A durian"
    ], [
        "A pillow with huskies printed on it",
        "A pillow"
    ], [
        "A DSLR photo of a wooden car, super detailed, best quality, 4K, HD.",
        "a wooden car."
]]
example_outputs_1 = [
    gr.Video(value=os.path.join(os.path.dirname(__file__), 'example/boots.mp4'), autoplay=True),
    gr.Video(value=os.path.join(os.path.dirname(__file__), 'example/Donut.mp4'), autoplay=True),
    gr.Video(value=os.path.join(os.path.dirname(__file__), 'example/durian.mp4'), autoplay=True),
    gr.Video(value=os.path.join(os.path.dirname(__file__), 'example/pillow_huskies.mp4'), autoplay=True),
    gr.Video(value=os.path.join(os.path.dirname(__file__), 'example/wooden_car.mp4'), autoplay=True)
]
example_outputs_2 = [
    gr.Video(value=os.path.join(os.path.dirname(__file__), 'example/boots_pro.mp4'), autoplay=True),
    gr.Video(value=os.path.join(os.path.dirname(__file__), 'example/Donut_pro.mp4'), autoplay=True),
    gr.Video(value=os.path.join(os.path.dirname(__file__), 'example/durian_pro.mp4'), autoplay=True),
    gr.Video(value=os.path.join(os.path.dirname(__file__), 'example/pillow_huskies_pro.mp4'), autoplay=True),
    gr.Video(value=os.path.join(os.path.dirname(__file__), 'example/wooden_car_pro.mp4'), autoplay=True)
]


def main(prompt, init_prompt, negative_prompt, num_iter, CFG, seed):
    if [prompt, init_prompt] in example_inputs:
        return example_outputs_1[example_inputs.index([prompt, init_prompt])], example_outputs_2[example_inputs.index([prompt, init_prompt])]
    args, lp, op, pp, gcp, gp = args_parser(default_opt=os.path.join(os.path.dirname(__file__), 'configs/white_hair_ironman.yaml'))
    gp.text = prompt
    gp.negative = negative_prompt
    if len(init_prompt) > 1: 
        gcp.init_shape = 'pointe' 
        gcp.init_prompt = init_prompt
    else:
        gcp.init_shape = 'sphere'
        gcp.init_prompt = '.'
    op.iterations = num_iter
    gp.guidance_scale = CFG
    gp.noise_seed = int(seed)
    print('==> User Prompt:', gp.text)
    lp.workspace = 'gradio_demo'
    video_path, pro_video_path = start_training(args, lp, op, pp, gcp, gp)
    return gr.Video(value=video_path, autoplay=True), gr.Video(value=pro_video_path, autoplay=True)

with gr.Blocks() as demo:
    gr.Markdown("# <center>LucidDreamer: Towards High-Fidelity Text-to-3D Generation via Interval Score Matching</center>")
    gr.Markdown("This live demo allows you to generate high-quality 3D content using text prompts. The outputs are 360° rendered 3d gaussian video and training progress visualization.<br> \
                It is based on Stable Diffusion 2.1. Please check out our <strong><a href=https://github.com/EnVision-Research/LucidDreamer>Project Page</a> / <a href=https://arxiv.org/abs/2311.11284>Paper</a> / <a href=https://github.com/EnVision-Research/LucidDreamer>Code</a></strong> if you want to learn more about our method!<br> \
                The running time might be longer than the reported 35 minutes (5000 iterations) on A100.<br> \
                &copy; This Gradio space was developed by Haodong LI.")
    gr.Interface(fn=main, inputs=[gr.Textbox(lines=2, value="A portrait of IRONMAN, white hair, head, photorealistic, 8K, HDR.", label="Your prompt"),
            gr.Textbox(lines=1, value="a man head.", label="Point-E init prompt (optional)"),
            gr.Textbox(lines=2, value="unrealistic, blurry, low quality, out of focus, ugly, low contrast, dull, low-resolution.", label="Negative prompt (optional)"),
            gr.Slider(1000, 5000, value=3000, label="Number of iterations"),
            gr.Slider(7.5, 100, value=7.5, label="CFG"),
            gr.Number(value=0, label="Seed")], 
        outputs=["playable_video", "playable_video"],
        examples=example_inputs,
        cache_examples=True,
        concurrency_limit=1)
demo.queue().launch(debug=True, share=True, inline=False)
