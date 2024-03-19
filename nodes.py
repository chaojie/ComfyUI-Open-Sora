import os
import sys
import folder_paths
comfy_path = os.path.dirname(folder_paths.__file__)
#sys.path.append(f'{comfy_path}/custom_nodes/ComfyUI-Open-Sora')
#print(sys.path)

from PIL import Image

import argparse
import torch
import numpy as np
import tempfile

#import colossalai
import torch.distributed as dist
from mmengine.runner import set_random_seed

from opensora.datasets import get_image
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.misc import to_torch_dtype
from opensora.acceleration.parallel_states import set_sequence_parallel_group
#from colossalai.cluster import DistCoordinator

pretrained_weights_path=f'{comfy_path}/models/checkpoints'
pretrained_weights=os.listdir(pretrained_weights_path)

config_path=f'{comfy_path}/custom_nodes/ComfyUI-Open-Sora/configs/opensora/inference'
config_lists=os.listdir(config_path)

class OpenSoraLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_path": (pretrained_weights, {"default": "OpenSora-v1-HQ-16x512x512.pth"}),
                "config": (config_lists, {"default": "16x512x512.py"}),
                "num_frames": ("INT", {"default": 16}),
                "width": ("INT", {"default": 512}),
                "height": ("INT", {"default": 512}),
                "dtype": ("STRING", {"default": "fp16"}),
                "num_sampling_steps": ("INT", {"default": 100}),
            },
        }

    RETURN_TYPES = ("MODEL","CLIP","VAE","SCHEDULER",)
    RETURN_NAMES = ("model","text_encoder","vae","scheduler",)
    FUNCTION = "run"
    CATEGORY = "OpenSora"

    def run(self,ckpt_path,config,num_frames,width,height,dtype,num_sampling_steps):
        ckpt_path=f'{pretrained_weights_path}/{ckpt_path}'
        config=f'{config_path}/{config}'

        # ======================================================
        # 1. cfg and init distributed env
        # ======================================================
        cfg = parse_configs(training=False,ckpt_path=ckpt_path,config=config)
        cfg.image_size=(width,height)
        print(cfg)

        # init distributed
        #colossalai.launch_from_torch({})
        #coordinator = DistCoordinator()

        #if coordinator.world_size > 1:
        #    set_sequence_parallel_group(dist.group.WORLD) 
        #    enable_sequence_parallelism = True
        #else:
        enable_sequence_parallelism = False

        # ======================================================
        # 2. runtime variables
        # ======================================================
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = to_torch_dtype(dtype)

        # ======================================================
        # 3. build model & load weights
        # ======================================================
        # 3.1. build model
        input_size = (num_frames, width,height)
        vae = build_module(cfg.vae, MODELS)
        latent_size = vae.get_latent_size(input_size)
        text_encoder = build_module(cfg.text_encoder, MODELS, device=device)  # T5 must be fp32
        model = build_module(
            cfg.model,
            MODELS,
            input_size=latent_size,
            in_channels=vae.out_channels,
            caption_channels=text_encoder.output_dim,
            model_max_length=text_encoder.model_max_length,
            dtype=dtype,
            enable_sequence_parallelism=enable_sequence_parallelism,
        )
        text_encoder.y_embedder = model.y_embedder  # hack for classifier-free guidance

        # 3.2. move to device & eval
        vae = vae.to(device, dtype).eval()
        model = model.to(device, dtype).eval()

        # 3.3. build scheduler
        cfg.scheduler["num_sampling_steps"] = num_sampling_steps
        scheduler = build_module(cfg.scheduler, SCHEDULERS)

        return (model,text_encoder,vae,scheduler,)

class OpenSoraSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "text_encoder": ("CLIP",),
                "vae": ("VAE",),
                "scheduler": ("SCHEDULER",),
                "prompt": ("STRING",{"default":""}),
                "dtype": ("STRING", {"default": "fp16"}),
                "seed": ("INT", {"default": 42}),
                "num_frames": ("INT", {"default": 16}),
                "width": ("INT", {"default": 512}),
                "height": ("INT", {"default": 512}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "run"
    CATEGORY = "OpenSora"

    def run(self,model,text_encoder,vae,scheduler,prompt,dtype,seed,num_frames,width,height):
        input_size = (num_frames, width,height)
        latent_size = vae.get_latent_size(input_size)
        set_random_seed(seed=seed)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = to_torch_dtype(dtype)

        # 3.4. support for multi-resolution
        model_args = dict()
        #if cfg.multi_resolution:
        #    image_size = cfg.image_size
        #    hw = torch.tensor([image_size], device=device, dtype=dtype).repeat(cfg.batch_size, 1)
        #    ar = torch.tensor([[image_size[0] / image_size[1]]], device=device, dtype=dtype).repeat(cfg.batch_size, 1)
        #    model_args["data_info"] = dict(ar=ar, hw=hw)
        
        batch_prompts=[prompt]

        text_encoder.t5.model.to('cuda').eval()
        model.to('cuda').eval()

        samples = scheduler.sample(
            model,
            text_encoder,
            z_size=(vae.out_channels, *latent_size),
            prompts=batch_prompts,
            device=device,
            additional_args=model_args,
        )

        text_encoder.t5.model.to('cpu')
        model.to('cpu')
        
        torch.cuda.empty_cache()

        return (samples.to(dtype),)

class OpenSoraRun:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "samples": ("LATENT",),
                "dtype": ("STRING", {"default": "fp16"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "OpenSora"

    def run(self,vae,samples,dtype):
        vae.to('cuda').eval()
        dtype = to_torch_dtype(dtype)
        samples = vae.decode(samples.to(dtype))
        vae.to('cpu')
        torch.cuda.empty_cache()
        
        outframes=[]
        for idx, sample in enumerate(samples):
            outframes=outframes+get_image(sample)

        return torch.cat(tuple(outframes), dim=0).unsqueeze(0)


NODE_CLASS_MAPPINGS = {
    "OpenSoraLoader":OpenSoraLoader,
    "OpenSoraSampler":OpenSoraSampler,
    "OpenSoraRun":OpenSoraRun,
}
