CUDA_VISIBLE_DEVICES="0,2,3" python examples/wanvideo/train_wan_t2v.py \
  --task data_process \
  --dataset_path data/dpo_data_12_sft_0511_with_r1_v0 \
  --output_path ./models \
  --text_encoder_path "Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth" \
  --vae_path "/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
  --tiled \
  --num_frames 81 \
  --height 480 \
  --width 832