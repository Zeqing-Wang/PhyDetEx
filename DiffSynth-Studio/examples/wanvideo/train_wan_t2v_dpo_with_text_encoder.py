import torch, os, imageio, argparse
from torchvision.transforms import v2
from einops import rearrange
import lightning as pl
import pandas as pd
from diffsynth import WanVideoPipeline, ModelManager, load_state_dict
from peft import LoraConfig, inject_adapter_in_model
import torchvision
from PIL import Image
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint

class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, max_num_frames=81, frame_interval=1, num_frames=81, height=480, width=832, is_i2v=False):
        metadata = pd.read_csv(metadata_path)
        self.path = [os.path.join(base_path, "train", file_name) for file_name in metadata["file_name"]]
        self.text = metadata["text"].to_list()
        
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.is_i2v = is_i2v
            
        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        
    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image


    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process):
        reader = imageio.get_reader(file_path)
        if reader.count_frames() < max_num_frames or reader.count_frames() - 1 < start_frame_id + (num_frames - 1) * interval:
            reader.close()
            return None
        
        frames = []
        first_frame = None
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            if first_frame is None:
                first_frame = np.array(frame)
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

        if self.is_i2v:
            return frames, first_frame
        else:
            return frames


    def load_video(self, file_path):
        start_frame_id = torch.randint(0, self.max_num_frames - (self.num_frames - 1) * self.frame_interval, (1,))[0]
        frames = self.load_frames_using_imageio(file_path, self.max_num_frames, start_frame_id, self.frame_interval, self.num_frames, self.frame_process)
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False
    
    
    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame)
        first_frame = frame
        frame = self.frame_process(frame)
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame


    def __getitem__(self, data_id):
        text = self.text[data_id]
        path = self.path[data_id]
        if self.is_image(path):
            if self.is_i2v:
                raise ValueError(f"{path} is not a video. I2V model doesn't support image-to-image training.")
            video = self.load_image(path)
        else:
            video = self.load_video(path)
        if self.is_i2v:
            video, first_frame = video
            data = {"text": text, "video": video, "path": path, "first_frame": first_frame}
        else:
            data = {"text": text, "video": video, "path": path}
        return data
    

    def __len__(self):
        return len(self.path)



class LightningModelForDataProcess(pl.LightningModule):
    def __init__(self, text_encoder_path, vae_path, image_encoder_path=None, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        super().__init__()
        model_path = [text_encoder_path, vae_path]
        if image_encoder_path is not None:
            model_path.append(image_encoder_path)
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models(model_path)
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)

        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        
    def test_step(self, batch, batch_idx):
        text, video, path = batch["text"][0], batch["video"], batch["path"][0]
        
        self.pipe.device = self.device
        if video is not None:
            # prompt
            prompt_emb = self.pipe.encode_prompt(text)
            # video
            video = video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            latents = self.pipe.encode_video(video, **self.tiler_kwargs)[0]
            # image
            if "first_frame" in batch:
                first_frame = Image.fromarray(batch["first_frame"][0].cpu().numpy())
                _, _, num_frames, height, width = video.shape
                image_emb = self.pipe.encode_image(first_frame, num_frames, height, width)
            else:
                image_emb = {}
            data = {"latents": latents, "prompt_emb": prompt_emb, "image_emb": image_emb}
            torch.save(data, path + ".tensors.pth")



class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, steps_per_epoch):
        metadata = pd.read_csv(metadata_path)
        self.path = [os.path.join(base_path, "train", file_name) for file_name in metadata["file_name"]]
        
        # WZQ Add 0414 添加label
        self.labels = metadata["label"].to_list() 
        self.labels = [label for i, label in enumerate(self.labels) if os.path.exists(self.path[i] + ".tensors.pth")]
        
        
        print(len(self.path), "videos in metadata.")
        self.path = [i + ".tensors.pth" for i in self.path if os.path.exists(i + ".tensors.pth")]
        print(len(self.path), "tensors cached in metadata.")
        assert len(self.path) > 0
        # print('part loaded tensor data:', self.path[:16])
        # assert False
        self.steps_per_epoch = steps_per_epoch


    def __getitem__(self, index):
        # print('index:', index)
        
        data_id = torch.randint(0, len(self.path), (1,))[0]
        data_id = (data_id + index) % len(self.path) # For fixed seed.
        # print('data id:', data_id)
        data_id = index
        # assert False
        path = self.path[data_id]
        data = torch.load(path, weights_only=True, map_location="cpu")
        
        # WZQ Add 0414
        label = self.labels[data_id]
        data["label"] = label
        data["file_name"] = self.path[data_id]
        
        return data
    

    def __len__(self):
        return self.steps_per_epoch



class LightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        dit_path,
        # WZQ 0414 Add Ref
        ref_model_path,
        # WZQ 0414 Add Ref
        # WZQ 0427 Add Loss Config
        loss_type,
        dpo_loss_factor,
        mse_loss_factor,
        # WZQ 0427 Add Loss Config
        learning_rate=1e-5,
        lora_rank=4, lora_alpha=4, train_architecture="lora", lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming",
        use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
        pretrained_lora_path=None
    ):
        super().__init__()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        if os.path.isfile(dit_path):
            model_manager.load_models([dit_path])
        else:
            dit_path = dit_path.split(",")
            model_manager.load_models([dit_path])
        
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.freeze_parameters()
        if train_architecture == "lora":
            self.add_lora_to_model(
                self.pipe.denoising_model(),
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_target_modules=lora_target_modules,
                init_lora_weights=init_lora_weights,
                pretrained_lora_path=pretrained_lora_path,
            )
        else:
            self.pipe.denoising_model().requires_grad_(True)
        
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        # WZQ 0414 加载参考模型
        self.load_ref_model(ref_model_path, lora_rank, lora_alpha, lora_target_modules, init_lora_weights, pretrained_lora_path)
        # WZQ 0427 添加loss config
        self.loss_type = loss_type
        self.dpo_loss_factor = dpo_loss_factor
        self.mse_loss_factor = mse_loss_factor
    # WZQ 0414 加载参考模型
    def load_ref_model(self, ref_model_path, lora_rank, lora_alpha, lora_target_modules, init_lora_weights, pretrained_lora_path):
        ref_model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        if os.path.isfile(ref_model_path):
            ref_model_manager.load_models([ref_model_path])
        else:
            ref_model_path = ref_model_path.split(",")
            ref_model_manager.load_models([ref_model_path])

        self.ref_pipe = WanVideoPipeline.from_model_manager(ref_model_manager)
        self.ref_pipe.scheduler.set_timesteps(1000, training=True)

        # 冻结参考模型参数
        self.ref_pipe.requires_grad_(False)
        self.ref_pipe.eval()

        # 为参考模型添加LoRA
        if "lora" in ref_model_path:
            self.add_lora_to_model(
                self.ref_pipe.denoising_model(),
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_target_modules=lora_target_modules,
                init_lora_weights=init_lora_weights,
                pretrained_lora_path=pretrained_lora_path,
            )
    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()
        # 0509 WZQ Add Text Model self.text_encoder
        self.pipe.text_embedding_model().train()
        
    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming", pretrained_lora_path=None, state_dict_converter=None):
        # Add LoRA to UNet
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True
            
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = inject_adapter_in_model(lora_config, model)
        for param in model.parameters():
            # Upcast LoRA parameters into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)
                
        # Lora pretrained lora weights
        if pretrained_lora_path is not None:
            state_dict = load_state_dict(pretrained_lora_path)
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            print(f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected.")
    

    def training_step(self, batch, batch_idx):
        # print(batch)
        # assert False
        print('----'*10,':',batch["file_name"])
        if batch["file_name"][0].replace('win','loss') != batch["file_name"][1]:
            print('Wrong DPO Sample in :', batch["file_name"])
            assert False
        # Data
        latents = batch["latents"].to(self.device)
        prompt_emb = batch["prompt_emb"]
        prompt_emb["context"] = prompt_emb["context"][0].to(self.device)
        image_emb = batch["image_emb"]
        if "clip_feature" in image_emb:
            image_emb["clip_feature"] = image_emb["clip_feature"][0].to(self.device)
        if "y" in image_emb:
            image_emb["y"] = image_emb["y"][0].to(self.device)

        # Loss
        self.pipe.device = self.device
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        extra_input = self.pipe.prepare_extra_input(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)


        # WZQ 0414 Bug is here!
        # print('prompt_emb:',prompt_emb)
        # print('extra_input:',extra_input)
        # print('image_emb:',image_emb)

        # print('noisy_latents shape:', noisy_latents.shape)
        # print('timestep shape:', timestep.shape)
        prompt_emb['context'] = prompt_emb['context'].repeat(noisy_latents.shape[0], 1, 1)
        # print('prompt_emb context shape:',prompt_emb['context'].shape)
        # print('extra_input type:',type(extra_input))
        # print('image_emb type:',type(image_emb))
        # print('prompt_emb type:',type(prompt_emb))
        # Compute loss
        noise_pred = self.pipe.denoising_model()(
            noisy_latents, timestep=timestep, **prompt_emb, **extra_input, **image_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
        )



        loss_mse = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        



        # WZQ 0414 添加ref model的输出
        # 让参考模型进行推理

        print('Begin Infering The Ref Model-----')
        self.ref_pipe.device = self.device
        ref_pred = self.ref_pipe.denoising_model()(
            noisy_latents, timestep=timestep, **prompt_emb, **extra_input, **image_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
        )
        print('End Infering The Ref Model-----')


        # WZQ 0414 计算DPO Loss
        model_losses = (noise_pred - training_target).pow(2).mean(dim=[1, 2, 3, 4])
        model_losses_w, model_losses_l = model_losses.chunk(2)
        ref_losses = (ref_pred - training_target).pow(2).mean(dim=[1, 2, 3, 4])
        ref_losses_w, ref_losses_l = ref_losses.chunk(2)
        model_diff = model_losses_w - model_losses_l
        ref_diff = ref_losses_w - ref_losses_l

        # DPO Loss 参数
        beta_dpo = 0.1
        dupfactor = 1.0

        # 计算 DPO Loss
        scale_term = -0.5 * beta_dpo
        inside_term = scale_term * (model_diff - ref_diff)
        implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
        import torch.nn.functional as F
        loss_DPO = (-1 * dupfactor * F.logsigmoid(inside_term)).mean()

        
        

        if self.loss_type == 'dpo_only':
            loss = loss_DPO
            print('DPO Loss:', loss.item())
            print('Implicit Accuracy:', implicit_acc.item())
        elif self.loss_type == 'mse_only':
            loss = loss_mse
            print('MSE Loss:', loss.item())
        else:
            loss = self.dpo_loss_factor * loss_DPO + self.mse_loss_factor * loss_mse
            print('MSE Loss:', loss_mse.item())
            print('DPO Loss:', loss_DPO.item())
            print('Total Loss:', loss.item())
            print('Implicit Accuracy:', implicit_acc.item())
    
        # Record log
        self.log("mse_loss", loss_DPO, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        self.log("dpo_loss", loss_mse, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        self.log("implicit_acc", implicit_acc, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        self.log("total_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)

        # loss = loss + loss_DPO
        # loss = loss_DPO
        loss = loss * self.pipe.scheduler.training_weight(timestep)


        return loss


    def configure_optimizers(self):
        # trainable_modules = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        
        # WZQ Add 0509 for text encoder
        dit_params = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        text_encoder_params = list(filter(lambda p: p.requires_grad, self.pipe.text_embedding_model().parameters()))
    
        # 联合所有可训练参数
        trainable_modules = dit_params + text_encoder_params


        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer
    

    # def on_save_checkpoint(self, checkpoint):
    #     checkpoint.clear()
    #     trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.denoising_model().named_parameters()))
    #     trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
    #     state_dict = self.pipe.denoising_model().state_dict()
    #     lora_state_dict = {}
    #     for name, param in state_dict.items():
    #         if name in trainable_param_names:
    #             lora_state_dict[name] = param
    #     checkpoint.update(lora_state_dict)

    # WZQ 0509
    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        
        # 收集DiT的可训练参数
        dit_trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, 
                                            self.pipe.denoising_model().named_parameters()))
        dit_trainable_param_names = set([named_param[0] for named_param in dit_trainable_param_names])
        
        # 收集text_encoder的可训练参数
        text_encoder_trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, 
                                                        self.pipe.text_embedding_model().named_parameters()))
        text_encoder_trainable_param_names = set([named_param[0] for named_param in text_encoder_trainable_param_names])
        
        # 合并所有可训练参数名称
        trainable_param_names = dit_trainable_param_names.union(text_encoder_trainable_param_names)
        
        # 保存所有可训练参数
        state_dict = {}
        # 添加DiT的参数
        for name, param in self.pipe.denoising_model().state_dict().items():
            if name in trainable_param_names:
                state_dict[f"denoising_model.{name}"] = param
        
        # 添加text_encoder的参数
        for name, param in self.pipe.text_embedding_model().state_dict().items():
            if name in trainable_param_names:
                state_dict[f"text_encoder.{name}"] = param
        
        checkpoint.update(state_dict)



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--task",
        type=str,
        default="data_process",
        required=True,
        choices=["data_process", "train"],
        help="Task. `data_process` or `train`.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default=None,
        help="Path of text encoder.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        help="Path of image encoder.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help="Path of VAE.",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default=None,
        help="Path of DiT.",
    )
    # WZQ Add 0414 ref model path
    parser.add_argument(
        "--ref_model_path",
        type=str,
        default=None,
        help="Path of reference model.",  # 添加参考模型路径参数
    )
    parser.add_argument(
        "--tiled",
        default=False,
        action="store_true",
        help="Whether enable tile encode in VAE. This option can reduce VRAM required.",
    )
    parser.add_argument(
        "--tile_size_height",
        type=int,
        default=34,
        help="Tile size (height) in VAE.",
    )
    parser.add_argument(
        "--tile_size_width",
        type=int,
        default=34,
        help="Tile size (width) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_height",
        type=int,
        default=18,
        help="Tile stride (height) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_width",
        type=int,
        default=16,
        help="Tile stride (width) in VAE.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=500,
        help="Number of steps per epoch.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Image width.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of batchsize.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q,k,v,o,ffn.0,ffn.2",
        help="Layers with LoRA modules.",
    )
    parser.add_argument(
        "--init_lora_weights",
        type=str,
        default="kaiming",
        choices=["gaussian", "kaiming"],
        help="The initializing method of LoRA weight.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="auto",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=4.0,
        help="The weight of the LoRA update matrices.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing_offload",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing offload.",
    )
    parser.add_argument(
        "--train_architecture",
        type=str,
        default="lora",
        choices=["lora", "full"],
        help="Model structure to train. LoRA training or full training.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        help="Whether Type Use The Loss Type.",
    )
    parser.add_argument(
        "--mse_loss_factor",
        type=float,
        help="factor for the mse loss.",
    )
    parser.add_argument(
        "--dpo_loss_factor",
        default=False,
        type=float,
        help="factor for the dpo loss.",
    )
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default=None,
        help="Pretrained LoRA path. Required if the training is resumed.",
    )
    parser.add_argument(
        "--use_swanlab",
        default=False,
        action="store_true",
        help="Whether to use SwanLab logger.",
    )
    parser.add_argument(
        "--swanlab_mode",
        default=None,
        help="SwanLab mode (cloud or local).",
    )
    args = parser.parse_args()
    return args


def data_process(args):
    dataset = TextVideoDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, "metadata.csv"),
        max_num_frames=args.num_frames,
        frame_interval=1,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        is_i2v=args.image_encoder_path is not None
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
        # WZQ Add 0414
        collate_fn=collate_fn
    )
    model = LightningModelForDataProcess(
        text_encoder_path=args.text_encoder_path,
        image_encoder_path=args.image_encoder_path,
        vae_path=args.vae_path,
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width),
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        default_root_dir=args.output_path,
    )
    trainer.test(model, dataloader)
    
    
def train(args):
    dataset = TensorDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, "metadata.csv"),
        steps_per_epoch=args.steps_per_epoch,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
        # WZQ Add 0414
        # collate_fn=collate_fn
    )
    model = LightningModelForTrain(
        dit_path=args.dit_path,
        # WZQ Add 0414
        ref_model_path=args.ref_model_path,
        learning_rate=args.learning_rate,
        train_architecture=args.train_architecture,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        init_lora_weights=args.init_lora_weights,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        pretrained_lora_path=args.pretrained_lora_path,
        dpo_loss_factor=args.dpo_loss_factor,
        mse_loss_factor=args.mse_loss_factor,
        loss_type=args.loss_type
    )
    if args.use_swanlab:
        from swanlab.integration.pytorch_lightning import SwanLabLogger
        swanlab_config = {"UPPERFRAMEWORK": "DiffSynth-Studio"}
        swanlab_config.update(vars(args))
        swanlab_logger = SwanLabLogger(
            project="wan", 
            name="wan",
            config=swanlab_config,
            mode=args.swanlab_mode,
            logdir=os.path.join(args.output_path, "swanlog"),
        )
        logger = [swanlab_logger]
    else:
        logger = None




    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16",
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        # accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1,every_n_train_steps=50,)],
        # callbacks=[checkpoint_callback],
        logger=logger,
        
    )
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    args = parse_args()
    if args.task == "data_process":
        data_process(args)
    elif args.task == "train":
        train(args)
