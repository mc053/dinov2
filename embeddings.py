import torch
import json
import argparse
from tqdm import tqdm
from pathlib import Path
from dinov2.utils.config import get_cfg_from_args
from dinov2.utils.utils import load_pretrained_weights
from dinov2.models import build_model, build_model_from_cfg
from dinov2.data import make_dataset, make_data_loader, DatasetWithEnumeratedTargets, SamplerType
from dinov2.data.datasets import CelebAOriginalVal
from dinov2.eval.utils import ModelWithNormalize, extract_features_with_dataloader
from dinov2.data.transforms import make_classification_eval_transform

class Args:
        def __init__(self, config_file, output_dir=None, opts=None):
            self.config_file = config_file
            self.output_dir = output_dir or ""
            self.opts = opts or []

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model")
    parser.add_argument("--input")

    args = parser.parse_args()

    model_name = args.model
    input_name = args.input

    args = Args(config_file=f"./{model_name}/config.yaml")
    cfg = get_cfg_from_args(args)
    model, _ = build_model_from_cfg(cfg, only_teacher=True)
    pretrained_weights = f'./{model_name}/eval/training_124999/teacher_checkpoint.pth'
    load_pretrained_weights(model, pretrained_weights, "teacher")
    model.eval()
    model.cuda()
    model = ModelWithNormalize(model) # better for comparing cosine similarity later.
    # we use by default the imagenet std and mean (as it is also used in the classification evals like knn)
    # for both CelebA and RVL-CDIP, since we're only interested in relative performance breakdowns.
    transform = make_classification_eval_transform()
    dataset = make_dataset(dataset_str=input_name, transform=transform)

    batch_size = 12
    results = {}
    output_file = f"./embeddings/{model_name}_{input_name}_emb.json"

    with torch.no_grad():
        batch_images = []
        batch_names = []

        for idx in tqdm(range(len(dataset)), desc="Processing images"):
            try:
                image, _ = dataset[idx]
                image_path = dataset.paths[idx]
                image_name = Path(image_path).name
    
                batch_images.append(image)
                batch_names.append(image_name)
    
                if len(batch_images) == batch_size or idx == len(dataset) - 1:
                    preprocessed_images = torch.stack([image for image in batch_images]).to("cuda")
                    embeddings = model.forward(preprocessed_images)
    
                    for name, embedding in zip(batch_names, embeddings):
                        results[name] = embedding.cpu().numpy().tolist()
    
                    batch_images = []
                    batch_names = []
    
            except Exception as e:
                print(f"Error processing image {idx}: {e}")
                continue
    
    with open(output_file, "w") as f:
        json.dump(results, f)
    
    print(f"Embeddings created and saved in {output_file}.")