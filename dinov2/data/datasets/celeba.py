import os
import pickle
import pandas as pd
import numpy as np
from .extended import ExtendedVisionDataset
from typing import Any, Callable, Optional, Tuple
from PIL import Image
from pathlib import Path
from enum import Enum

class Target(Enum):
    IDENTITY = 1
    GENDER = 2

# Set this value before performing evaluation.
TARGET = Target.IDENTITY

class CelebAOriginalTrain(ExtendedVisionDataset):
    def __init__(self, root: str = os.path.dirname(os.path.abspath(__file__)), transforms=None,
    transform=None, target_transform=None, image_dir_name="CelebA_original/train"):
        super().__init__(root=root, transforms=transforms, transform=transform, target_transform=target_transform)
        self.root = Path(root).resolve()
        self.image_dir = self.root / "CelebA" / image_dir_name
        self.photo_files_path = self.image_dir / "image_list.pickle"
        self.paths = []
        self._load_image_paths()
        identity_file_path = self.root / "CelebA" / "identity_CelebA.txt"
        attr_file_path = self.root / "CelebA" / "list_attr_celeba.csv"
        self.identity_map = pd.read_csv(identity_file_path, sep=" ", header=None, names=["image_id", "identity"], index_col="image_id").to_dict()["identity"]
        self.gender_map = pd.read_csv(attr_file_path, index_col="image_id")["Male"].to_dict()

    def _load_image_paths(self):
        if self.photo_files_path.exists():
            print("Load image list")
            with open(self.photo_files_path, "rb") as handle:
                self.paths = pickle.load(handle)
        else:
            print("Creating image list")
            for img_path in self.image_dir.glob("*.jpg"):
                self.paths.append(img_path)
            with open(self.photo_files_path, "wb") as handle:
                pickle.dump(self.paths, handle)

    def get_image_data(self, index: int) -> bytes:
        image_path = self.paths[index]
        image = Image.open(image_path).convert(mode="RGB")
        return image

    def get_target(self, index: int) -> Any:
        try:
            image_name = os.path.basename(self.paths[index])
            if TARGET == Target.IDENTITY:
                return self.identity_map[image_name]
            elif TARGET == Target.GENDER:
                gender = self.gender_map[image_name]
                return 0 if gender == -1 else gender # otherwise the dino evaluation framework throws an exception: "assert torch.all(all_labels > -1)"
            else:
                raise ValueError(f"Unsupported target type: {TARGET}")
        except KeyError as e:
            raise RuntimeError(f"Target for image {os.path.basename(self.paths[index])} not found") from e

    def get_targets(self) -> Optional[np.ndarray]:
        if TARGET == Target.IDENTITY:
            return np.array([self.identity_map[os.path.basename(path)] for path in self.paths])
        elif TARGET == Target.GENDER:
            return np.array([0 if self.gender_map[os.path.basename(path)] == -1 else self.gender_map[os.path.basename(path)] for path in self.paths])
        else:
            raise ValueError(f"Unsupported target type: {TARGET}")

    def __getitem__(self, index):
        try:
            image = self.get_image_data(index)
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e
        target = self.get_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.paths)

class CelebAOriginalVal(CelebAOriginalTrain):
    def __init__(self, root: str = os.path.dirname(os.path.abspath(__file__)), transforms=None,
    transform=None, target_transform=None, image_dir_name="CelebA_original/val"):
        super().__init__(root=root, transforms=transforms, transform=transform,
        target_transform=target_transform, image_dir_name=image_dir_name)

class CelebAPixelatedTrain(CelebAOriginalTrain):
    def __init__(self, root: str = os.path.dirname(os.path.abspath(__file__)), transforms=None,
    transform=None, target_transform=None, image_dir_name="CelebA_pixelated/train"):
        super().__init__(root=root, transforms=transforms, transform=transform,
        target_transform=target_transform, image_dir_name=image_dir_name)

class CelebAMaskedTrain(CelebAOriginalTrain):
    def __init__(self, root: str = os.path.dirname(os.path.abspath(__file__)), transforms=None,
    transform=None, target_transform=None, image_dir_name="CelebA_masked/train"):
        super().__init__(root=root, transforms=transforms, transform=transform,
        target_transform=target_transform, image_dir_name=image_dir_name)

# def test():
#     # dataset_path = "/home/stud/m/mc085/mounted_home/dinov2/dinov2/data/datasets/CelebA/CelebA_original/train/"
#     dataset = CelebAOriginalVal()
#     print(f"No. of pictures in dataset: {len(dataset)}")
#     print(f"Targets in dataset: {dataset.get_targets()}")
#     
#     # dataset_file_names = [os.path.basename(path) for path in dataset.paths]
#     # 
#     # test_cases = ['000103.jpg', '002822.jpg', '005864.jpg', '013191.jpg', '015069.jpg', '024891.jpg', '030232.jpg', '040038.jpg', '040972.jpg', '041216.jpg', '045871.jpg', '051897.jpg', '070373.jpg', '086045.jpg', '086926.jpg', '090258.jpg', '091632.jpg', '094740.jpg', '106949.jpg', '112659.jpg', '113173.jpg', '125945.jpg', '128046.jpg', '133552.jpg', '137651.jpg', '138272.jpg', '138569.jpg', '144857.jpg', '152937.jpg', '158023.jpg']
#     # for img_name in test_cases:
#     #     try:
#     #         index = dataset_file_names.index(img_name)
#     #         target = dataset.get_target(index)
#     #         print(f"Index: {index}, Image: {img_name}, Target: {target}")
#     #     except ValueError:
#     #         print(f"Image: {img_name}, Error: '{img_name}' is not in dataset")
#     #     except Exception as e:
#     #         print(f"Image: {img_name}, Error: {e}")
# 
# if __name__ == "__main__":
#     test()

## class CelebABlurred(CelebAOriginal) ...
## class CelebADistorted(CelebAOriginal) ...