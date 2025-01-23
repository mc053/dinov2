# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Taken from https://gitlab.com/ziegleto-machine-learning/dino/-/blob/main/visualize_attention.py

import os
import sys
import argparse
import random
import colorsys
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
from embeddings import Args
import numpy as np
from PIL import Image
from dinov2.models.vision_transformer import vit_small, vit_large
from dinov2.utils.config import get_cfg_from_args
from dinov2.utils.utils import load_pretrained_weights
from dinov2.models import build_model_from_cfg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--image")

    args = parser.parse_args()
    
    model_name = args.model
    image_name = args.image

    patch_size = 16 # all trained models are ViTs (large) with 16x16 patch sizes.
    image_size = (1600, 1600) # HxW must be a multiple of 16.
    output_dir = f'./{model_name}/eval/training_124999/attention_visualization/{image_name}'
    args = Args(config_file=f"./{model_name}/config.yaml")
    cfg = get_cfg_from_args(args)
    model, _ = build_model_from_cfg(cfg, only_teacher=True)
    pretrained_weights = f'./{model_name}/eval/training_124999/teacher_checkpoint.pth'
    load_pretrained_weights(model, pretrained_weights, "teacher")

    for p in model.parameters():
        p.requires_grad = False

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
    model.to(device)
    model.eval()

    img = Image.open(f'./{image_name}')
    img = img.convert('RGB')
    transform = pth_transforms.Compose([
        pth_transforms.Resize(image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = transform(img)

    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0)
 
    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    attentions = model.get_last_self_attention(img.to(device))
    nh = attentions.shape[1] # number of head
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

    os.makedirs(output_dir, exist_ok=True)

    for j in range(nh):
        fname = os.path.join(output_dir, "attn-head" + str(j) + ".png")
        plt.imsave(fname=fname, arr=attentions[j], format='png')
        print(f"{fname} saved.")