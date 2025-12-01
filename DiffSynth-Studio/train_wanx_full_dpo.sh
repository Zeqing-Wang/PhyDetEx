CUDA_VISIBLE_DEVICES="2" python examples/wanvideo/train_wan_t2v_dpo.py \
  --task train \
  --train_architecture full \
  --dataset_path data/dpo_data_12_sft_0328 \
  --metadata_file metadata_refine.csv \
  --output_path ./finetuned_models/wanx/models_example_dpo_data_12_sft_0328_1_e6_dpo_with_mse_d_1_m_0_5_with_refine_prompts \
  --dit_path "/home/notebook/data/personal/S9057536/models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
  --steps_per_epoch 2500 \
  --max_epochs 1 \
  --learning_rate 1e-6 \
  --accumulate_grad_batches 1 \
  --use_gradient_checkpointing \
  --ref_model_path "/home/notebook/data/personal/S9057536/models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
  --batch_size 2 \
  --dpo_loss_factor 1 \
  --mse_loss_factor 0.5 \
  --save_freq 250 \
  --loss_type 'dpo_with_mse' # dpo_only mse_only dpo_with_mse


  # 2707*2 = 5414