# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import random

from PIL import Image
import torchvision.transforms as T
import os

def collate_data_and_cast(samples_list, mask_ratio_tuple, mask_probability, dtype, n_tokens=None, mask_generator=None):
    # dtype = torch.half  # TODO: Remove

    n_global_crops = len(samples_list[0][0]["global_crops"])
    n_local_crops = len(samples_list[0][0]["local_crops"])

    collated_global_crops = torch.stack([s[0]["global_crops"][i] for i in range(n_global_crops) for s in samples_list])

    collated_local_crops = torch.stack([s[0]["local_crops"][i] for i in range(n_local_crops) for s in samples_list])

    # Visualize crops
    transform_to_image = T.ToPILImage()
    output_dir = "crop_visualizations"
    os.makedirs(output_dir, exist_ok=True)

    if any(os.listdir(output_dir)):  # Save crops of only one example.
        print(f"Directory '{output_dir}' is not empty. Skipping crop visualization.")
    else:
        for sample in samples_list:
            # Save all global crops
            global_crops = sample[0]["global_crops"]
            for crop_idx, crop in enumerate(global_crops):
                global_crop_img = transform_to_image(crop.cpu())
                filename = os.path.join(output_dir, f"global_crop{crop_idx}.png")
                global_crop_img.save(filename)
                print(f"Saved: {filename}")

            # Save all local crops
            local_crops = sample[1]["local_crops"]
            for crop_idx, crop in enumerate(local_crops):
                local_crop_img = transform_to_image(crop.cpu())
                filename = os.path.join(output_dir, f"local_crop{crop_idx}.png")
                local_crop_img.save(filename)
                print(f"Saved: {filename}")

    B = len(collated_global_crops)
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
        upperbound += int(N * prob_max)
    for i in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    return {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }
