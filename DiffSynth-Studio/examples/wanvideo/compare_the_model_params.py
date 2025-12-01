import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
from modelscope import snapshot_download
import pandas as pd
import os
from tqdm import tqdm
import random

def get_caption_list_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df['caption'].tolist()

def compare_models(original_model, trained_model):
    # Get state dictionaries
    original_state_dict = original_model.state_dict()
    trained_state_dict = trained_model.state_dict()
    
    # Compare parameter shapes and values
    for key in original_state_dict:
        if key in trained_state_dict:
            # Check if shapes match
            if original_state_dict[key].shape != trained_state_dict[key].shape:
                print(f"Shape mismatch for {key}: original {original_state_dict[key].shape} vs trained {trained_state_dict[key].shape}")
                continue
            
            # Check if values are exactly the same
            if torch.equal(original_state_dict[key], trained_state_dict[key]):
                print(f"Parameters identical for {key}")
            else:
                print(f"Parameters differ for {key}")
                # Calculate difference magnitude
                diff = torch.abs(original_state_dict[key] - trained_state_dict[key])
                print(f"Max difference: {diff.max().item()}, Mean difference: {diff.mean().item()}")
        else:
            print(f"Key {key} not found in trained model")
    
    # Check for any extra keys in trained model
    for key in trained_state_dict:
        if key not in original_state_dict:
            print(f"Extra key in trained model: {key}")

if __name__ == '__main__':
    # Original and trained model paths
    original_unet_path = "/home/notebook/data/personal/S9057536/models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"
    trained_unet_path = '/home/notebook/code/personal/S9057536/ExpVideoAbnormal_Aliyun/DiffSynth-Studio/models_example_dpo_data_12_sft_0328_1_e5/lightning_logs/version_1/checkpoints/epoch=0-step=1250.ckpt'
    
    # Load both models
    print("Loading original model...")
    original_manager = ModelManager(device="cpu")
    original_manager.load_models(
        [
            original_unet_path,
            "/home/notebook/data/personal/S9057536/models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
            "/home/notebook/data/personal/S9057536/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
        ],
        torch_dtype=torch.bfloat16
    )
    
    print("Loading trained model...")
    trained_manager = ModelManager(device="cpu")
    trained_manager.load_models(
        [
            trained_unet_path,
            "/home/notebook/data/personal/S9057536/models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
            "/home/notebook/data/personal/S9057536/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
        ],
        torch_dtype=torch.bfloat16
    )
    
    # Compare the UNet parts of the models
    print("\nComparing UNet parameters...")
    compare_models(original_manager.unet, trained_manager.unet)
    
    # Optionally compare other components if needed
    # print("\nComparing text encoder parameters...")
    # compare_models(original_manager.text_encoder, trained_manager.text_encoder)
    
    # print("\nComparing VAE parameters...")
    # compare_models(original_manager.vae, trained_manager.vae)