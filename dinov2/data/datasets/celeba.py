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
TARGET = Target.GENDER

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

class CelebAPixelatedVal(CelebAOriginalTrain):
    def __init__(self, root: str = os.path.dirname(os.path.abspath(__file__)), transforms=None,
    transform=None, target_transform=None, image_dir_name="CelebA_pixelated/val"):
        super().__init__(root=root, transforms=transforms, transform=transform,
        target_transform=target_transform, image_dir_name=image_dir_name)

class CelebAMaskedTrain(CelebAOriginalTrain):
    def __init__(self, root: str = os.path.dirname(os.path.abspath(__file__)), transforms=None,
    transform=None, target_transform=None, image_dir_name="CelebA_masked/train"):
        super().__init__(root=root, transforms=transforms, transform=transform,
        target_transform=target_transform, image_dir_name=image_dir_name)

class CelebAMaskedVal(CelebAOriginalTrain):
    def __init__(self, root: str = os.path.dirname(os.path.abspath(__file__)), transforms=None,
    transform=None, target_transform=None, image_dir_name="CelebA_masked/val"):
        super().__init__(root=root, transforms=transforms, transform=transform,
        target_transform=target_transform, image_dir_name=image_dir_name)

class CelebABlurredTrain(CelebAOriginalTrain):
    def __init__(self, root: str = os.path.dirname(os.path.abspath(__file__)), transforms=None,
    transform=None, target_transform=None, image_dir_name="CelebA_blurred/train"):
        super().__init__(root=root, transforms=transforms, transform=transform,
        target_transform=target_transform, image_dir_name=image_dir_name)

class CelebABlurredVal(CelebAOriginalTrain):
    def __init__(self, root: str = os.path.dirname(os.path.abspath(__file__)), transforms=None,
    transform=None, target_transform=None, image_dir_name="CelebA_blurred/val"):
        super().__init__(root=root, transforms=transforms, transform=transform,
        target_transform=target_transform, image_dir_name=image_dir_name)

class CelebAABTrain(ExtendedVisionDataset):
    def __init__(self, root: str = os.path.dirname(os.path.abspath(__file__)), transforms=None,
                 transform=None, target_transform=None, 
                 original_dir_name="CelebA_original/train", 
                 anonymized_dir_name=None):
        super().__init__(root=root, transforms=transforms, transform=transform, target_transform=target_transform)
        if anonymized_dir_name is None:
            raise NotImplementedError()
        self.root = Path(root).resolve()
        self.original_image_dir = self.root / "CelebA" / original_dir_name
        self.anonymized_image_dir = self.root / "CelebA" / anonymized_dir_name
        self.photo_files_path = self.original_image_dir / f"image_list_{Path(anonymized_dir_name).parent.name}.pickle"
        self.paths = []
        self._load_image_paths()
        identity_file_path = self.root / "CelebA" / "identity_CelebA.txt"
        attr_file_path = self.root / "CelebA" / "list_attr_celeba.csv"
        self.identity_map = pd.read_csv(identity_file_path, sep=" ", header=None, 
                                        names=["image_id", "identity"], index_col="image_id").to_dict()["identity"]
        self.gender_map = pd.read_csv(attr_file_path, index_col="image_id")["Male"].to_dict()

    def _load_image_paths(self):
        if self.photo_files_path.exists():
            print("Load image list")
            with open(self.photo_files_path, "rb") as handle:
                self.paths = pickle.load(handle)
        else:
            print("Creating image list")
            for img_path in self.original_image_dir.glob("*.jpg"):
                self.paths.append(img_path.name)  # Save just the image name
            with open(self.photo_files_path, "wb") as handle:
                pickle.dump(self.paths, handle)

    def get_image_data(self, image_name: str, directory: Path) -> bytes:
        image_path = directory / image_name
        image = Image.open(image_path).convert(mode="RGB")
        return image

    def get_target(self, image_name: str) -> Any:
        try:
            if TARGET == Target.IDENTITY:
                return self.identity_map[image_name]
            elif TARGET == Target.GENDER:
                gender = self.gender_map[image_name]
                return 0 if gender == -1 else gender
            else:
                raise ValueError(f"Unsupported target type: {TARGET}")
        except KeyError as e:
            raise RuntimeError(f"Target for image {image_name} not found") from e

    def get_targets(self) -> Optional[np.ndarray]:
        if TARGET == Target.IDENTITY:
            return np.array([self.identity_map[path] for path in self.paths])
        elif TARGET == Target.GENDER:
            return np.array([0 if self.gender_map[path] == -1 else self.gender_map[path] for path in self.paths])
        else:
            raise ValueError(f"Unsupported target type: {TARGET}")

    def __getitem__(self, index):
        try:
            image_name = self.paths[index]
            original_image = self.get_image_data(image_name, self.original_image_dir)
            anonymized_image = self.get_image_data(image_name, self.anonymized_image_dir)
        except Exception as e:
            raise RuntimeError(f"Cannot read image for sample {index}") from e

        target = self.get_target(image_name)

        if self.transforms is not None:
            original_image, target = self.transforms(original_image, target)
            anonymized_image, _ = self.transforms(anonymized_image, target)  # Apply the same transforms to anonymized image

        return original_image, anonymized_image, target

    def __len__(self):
        return len(self.paths)

class CelebAPixelatedABTrain(CelebAABTrain):
    def __init__(self, root: str = os.path.dirname(os.path.abspath(__file__)), transforms=None,
                 transform=None, target_transform=None):
        super().__init__(root=root, transforms=transforms, transform=transform, target_transform=target_transform,
                         anonymized_dir_name="CelebA_pixelated/train")

class CelebAMaskedABTrain(CelebAABTrain):
    def __init__(self, root: str = os.path.dirname(os.path.abspath(__file__)), transforms=None,
                 transform=None, target_transform=None):
        super().__init__(root=root, transforms=transforms, transform=transform, target_transform=target_transform,
                         anonymized_dir_name="CelebA_masked/train")

class CelebABlurredABTrain(CelebAABTrain):
    def __init__(self, root: str = os.path.dirname(os.path.abspath(__file__)), transforms=None,
                 transform=None, target_transform=None):
        super().__init__(root=root, transforms=transforms, transform=transform, target_transform=target_transform,
                         anonymized_dir_name="CelebA_blurred/train")