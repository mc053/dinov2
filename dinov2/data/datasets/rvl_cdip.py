import os
import pickle
import numpy as np
from .extended import ExtendedVisionDataset
from typing import Any, Optional
from PIL import Image
from pathlib import Path

class RvlCdipOriginalTrain(ExtendedVisionDataset):
    def __init__(self, root: str = os.path.dirname(os.path.abspath(__file__)), transforms=None,
    transform=None, target_transform=None, image_dir_name="RVL-CDIP_original/train"):
        super().__init__(root=root, transforms=transforms, transform=transform, target_transform=target_transform)
        self.root = Path(root).resolve()
        self.image_dir = self.root / "RVL-CDIP" / image_dir_name
        self.photo_files_path = self.image_dir / "image_list.pickle"
        self.paths = []
        self._load_image_paths()

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
            label_str = image_name.split("_label_")[-1].split(".")[0]
            return int(label_str)
        except (IndexError, ValueError) as e:
            raise RuntimeError(f"Couldn't extract target for {image_name}") from e

    def get_targets(self) -> Optional[np.ndarray]:
        return np.array([self.get_target(idx) for idx in range(len(self.paths))])

    def __getitem__(self, index: int):
        try:
            image = self.get_image_data(index)
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e
        target = self.get_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.paths)

class RvlCdipOriginalVal(RvlCdipOriginalTrain):
    def __init__(self, root: str = os.path.dirname(os.path.abspath(__file__)), transforms=None,
    transform=None, target_transform=None, image_dir_name="RVL-CDIP_original/val"):
        super().__init__(root=root, transforms=transforms, transform=transform,
        target_transform=target_transform, image_dir_name=image_dir_name)

class RvlCdip100MaskedTrain(RvlCdipOriginalTrain):
    def __init__(self, root: str = os.path.dirname(os.path.abspath(__file__)), transforms=None,
    transform=None, target_transform=None, image_dir_name="RVL-CDIP_100_masked/train"):
        super().__init__(root=root, transforms=transforms, transform=transform,
        target_transform=target_transform, image_dir_name=image_dir_name)

class RvlCdip100MaskedVal(RvlCdipOriginalTrain):
    def __init__(self, root: str = os.path.dirname(os.path.abspath(__file__)), transforms=None,
    transform=None, target_transform=None, image_dir_name="RVL-CDIP_100_masked/val"):
        super().__init__(root=root, transforms=transforms, transform=transform,
        target_transform=target_transform, image_dir_name=image_dir_name)

class RvlCdip100PixelatedTrain(RvlCdipOriginalTrain):
    def __init__(self, root: str = os.path.dirname(os.path.abspath(__file__)), transforms=None,
    transform=None, target_transform=None, image_dir_name="RVL-CDIP_100_pixelated/train"):
        super().__init__(root=root, transforms=transforms, transform=transform,
        target_transform=target_transform, image_dir_name=image_dir_name)

class RvlCdip100PixelatedVal(RvlCdipOriginalTrain):
    def __init__(self, root: str = os.path.dirname(os.path.abspath(__file__)), transforms=None,
    transform=None, target_transform=None, image_dir_name="RVL-CDIP_100_pixelated/val"):
        super().__init__(root=root, transforms=transforms, transform=transform,
        target_transform=target_transform, image_dir_name=image_dir_name)

class RvlCdip100BlurredTrain(RvlCdipOriginalTrain):
    def __init__(self, root: str = os.path.dirname(os.path.abspath(__file__)), transforms=None,
    transform=None, target_transform=None, image_dir_name="RVL-CDIP_100_blurred/train"):
        super().__init__(root=root, transforms=transforms, transform=transform,
        target_transform=target_transform, image_dir_name=image_dir_name)

class RvlCdip100BlurredVal(RvlCdipOriginalTrain):
    def __init__(self, root: str = os.path.dirname(os.path.abspath(__file__)), transforms=None,
    transform=None, target_transform=None, image_dir_name="RVL-CDIP_100_blurred/val"):
        super().__init__(root=root, transforms=transforms, transform=transform,
        target_transform=target_transform, image_dir_name=image_dir_name)

class RvlCdip50MaskedTrain(RvlCdipOriginalTrain):
    def __init__(self, root: str = os.path.dirname(os.path.abspath(__file__)), transforms=None,
    transform=None, target_transform=None, image_dir_name="RVL-CDIP_50_masked/train"):
        super().__init__(root=root, transforms=transforms, transform=transform,
        target_transform=target_transform, image_dir_name=image_dir_name)

class RvlCdip50MaskedVal(RvlCdipOriginalTrain):
    def __init__(self, root: str = os.path.dirname(os.path.abspath(__file__)), transforms=None,
    transform=None, target_transform=None, image_dir_name="RVL-CDIP_50_masked/val"):
        super().__init__(root=root, transforms=transforms, transform=transform,
        target_transform=target_transform, image_dir_name=image_dir_name)

class RvlCdip50PixelatedTrain(RvlCdipOriginalTrain):
    def __init__(self, root: str = os.path.dirname(os.path.abspath(__file__)), transforms=None,
    transform=None, target_transform=None, image_dir_name="RVL-CDIP_50_pixelated/train"):
        super().__init__(root=root, transforms=transforms, transform=transform,
        target_transform=target_transform, image_dir_name=image_dir_name)

class RvlCdip50PixelatedVal(RvlCdipOriginalTrain):
    def __init__(self, root: str = os.path.dirname(os.path.abspath(__file__)), transforms=None,
    transform=None, target_transform=None, image_dir_name="RVL-CDIP_50_pixelated/val"):
        super().__init__(root=root, transforms=transforms, transform=transform,
        target_transform=target_transform, image_dir_name=image_dir_name)

class RvlCdip50BlurredTrain(RvlCdipOriginalTrain):
    def __init__(self, root: str = os.path.dirname(os.path.abspath(__file__)), transforms=None,
    transform=None, target_transform=None, image_dir_name="RVL-CDIP_50_blurred/train"):
        super().__init__(root=root, transforms=transforms, transform=transform,
        target_transform=target_transform, image_dir_name=image_dir_name)

class RvlCdip50BlurredVal(RvlCdipOriginalTrain):
    def __init__(self, root: str = os.path.dirname(os.path.abspath(__file__)), transforms=None,
    transform=None, target_transform=None, image_dir_name="RVL-CDIP_50_blurred/val"):
        super().__init__(root=root, transforms=transforms, transform=transform,
        target_transform=target_transform, image_dir_name=image_dir_name)

class RvlCdipABTrain(ExtendedVisionDataset):
    def __init__(self, root: str = os.path.dirname(os.path.abspath(__file__)), transforms=None,
                 transform=None, target_transform=None,
                 original_dir_name="RVL-CDIP_original/train",
                 anonymized_dir_name=None):
        super().__init__(root=root, transforms=transforms, transform=transform, target_transform=target_transform)
        if anonymized_dir_name is None:
            raise NotImplementedError()
        self.root = Path(root).resolve()
        self.original_image_dir = self.root / "RVL-CDIP" / original_dir_name
        self.anonymized_image_dir = self.root / "RVL-CDIP" / anonymized_dir_name
        self.photo_files_path = self.original_image_dir / f"image_list_{Path(anonymized_dir_name).parent.name}.pickle"
        self.paths = []
        self._load_image_paths()

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

    def get_target(self, image_name: str) -> int:
        try:
            label_str = image_name.split("_label_")[-1].split(".")[0]
            return int(label_str)
        except (IndexError, ValueError) as e:
            raise RuntimeError(f"Couldn't extract target for {image_name}") from e

    def get_targets(self) -> np.ndarray:
        return np.array([self.get_target(path) for path in self.paths])

    def __getitem__(self, index: int):
        try:
            image_name = self.paths[index]
            original_image = self.get_image_data(image_name, self.original_image_dir)
            anonymized_image = self.get_image_data(image_name, self.anonymized_image_dir)
        except Exception as e:
            raise RuntimeError(f"Cannot read image for sample {index}: {e}")

        target = self.get_target(image_name)

        if self.transforms is not None:
            original_image, target = self.transforms(original_image, target)
            anonymized_image, _ = self.transforms(anonymized_image, target) # Apply the same transforms to anonymized image

        return original_image, anonymized_image, target

    def __len__(self):
        return len(self.paths)

class RvlCdip100PixelatedABTrain(RvlCdipABTrain):
    def __init__(self, root: str = os.path.dirname(os.path.abspath(__file__)), transforms=None,
                 transform=None, target_transform=None):
        super().__init__(root=root, transforms=transforms, transform=transform, target_transform=target_transform,
                         anonymized_dir_name="RVL-CDIP_100_pixelated/train")

class RvlCdip100MaskedABTrain(RvlCdipABTrain):
    def __init__(self, root: str = os.path.dirname(os.path.abspath(__file__)), transforms=None,
                 transform=None, target_transform=None):
        super().__init__(root=root, transforms=transforms, transform=transform, target_transform=target_transform,
                         anonymized_dir_name="RVL-CDIP_100_masked/train")

class RvlCdip100BlurredABTrain(RvlCdipABTrain):
    def __init__(self, root: str = os.path.dirname(os.path.abspath(__file__)), transforms=None,
                 transform=None, target_transform=None):
        super().__init__(root=root, transforms=transforms, transform=transform, target_transform=target_transform,
                         anonymized_dir_name="RVL-CDIP_100_blurred/train")

class RvlCdip50MaskedABTrain(RvlCdipABTrain):
    def __init__(self, root: str = os.path.dirname(os.path.abspath(__file__)), transforms=None,
                 transform=None, target_transform=None):
        super().__init__(root=root, transforms=transforms, transform=transform, target_transform=target_transform,
                         anonymized_dir_name="RVL-CDIP_50_masked/train")

class RvlCdip50PixelatedABTrain(RvlCdipABTrain):
    def __init__(self, root: str = os.path.dirname(os.path.abspath(__file__)), transforms=None,
                 transform=None, target_transform=None):
        super().__init__(root=root, transforms=transforms, transform=transform, target_transform=target_transform,
                         anonymized_dir_name="RVL-CDIP_50_pixelated/train")

class RvlCdip50BlurredABTrain(RvlCdipABTrain):
    def __init__(self, root: str = os.path.dirname(os.path.abspath(__file__)), transforms=None,
                 transform=None, target_transform=None):
        super().__init__(root=root, transforms=transforms, transform=transform, target_transform=target_transform,
                         anonymized_dir_name="RVL-CDIP_50_blurred/train")