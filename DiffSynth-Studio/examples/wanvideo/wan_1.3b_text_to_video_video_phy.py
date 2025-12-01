import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
from modelscope import snapshot_download
import pandas as pd
import os
from tqdm import tqdm
import random
# Download models
# /home/notebook/data/personal/S9057536/models/Wan2.1-T2V-1.3B
# snapshot_download("Wan-AI/Wan2.1-T2V-1.3B", local_dir="models/Wan-AI/Wan2.1-T2V-1.3B")

# Load models

def get_caption_list_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df['caption'].tolist()


if __name__ == '__main__':
    file_path = '/home/notebook/code/personal/S9057536/ExpVideoAbnormal_Aliyun/videophy_eval/videophy_test_public_cogvideo.csv'
    # file_path = '/home/notebook/code/personal/S9057536/ExpVideoAbnormal_Aliyun/videophy2_eval/unique_captions.csv'
    output_path = '/home/notebook/code/personal/S9057536/ExpVideoAbnormal_Aliyun/videophy1_eval/models_example_dpo_data_12_vlm_r1_v0_0511_6_e6_all_dpo_with_mse_d_1_m_0_5_loss_des_0.1_m1_fix/video/epoch=0-step=2500'
    # Ori: "/home/notebook/data/personal/S9057536/models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"
    unet_path = '/home/notebook/code/personal/S9057536/ExpVideoAbnormal_Aliyun/DiffSynth-Studio/finetuned_models/wanx/models_example_dpo_data_12_vlm_r1_v0_0511_6_e6_all_dpo_with_mse_d_1_m_0_5_loss_des_0.1_m1_fix/lightning_logs/version_0/checkpoints/epoch=0-step=2500.ckpt'

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # file_path = '/home/notebook/code/personal/S9057536/ExpVideoAbnormal_Aliyun/videophy2_eval/unique_captions.csv'
    # output_path = '/home/notebook/code/personal/S9057536/ExpVideoAbnormal_Aliyun/videophy2_eval/wanx_videophy_1_3b_base'

    caption_list = get_caption_list_from_csv(file_path=file_path)
    model_manager = ModelManager(device="cpu")
    
    model_manager.load_models(
        [
            # "/home/notebook/data/personal/S9057536/models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
            unet_path,
            "/home/notebook/data/personal/S9057536/models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
            "/home/notebook/data/personal/S9057536/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
        ],
        torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
    )
    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    random.shuffle(caption_list)
    for caption in tqdm(caption_list):
        video_name = caption.replace(' ','_')+'.mp4'
        output_video_path = os.path.join(output_path, video_name)
        if os.path.exists(output_video_path):
            print('skip:', output_video_path)
            continue
        # Text-to-video
        video = pipe(
            prompt=caption,
            negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            num_inference_steps=50,
            seed=0, tiled=True
        )
        save_video(video, output_video_path, fps=15, quality=5)
