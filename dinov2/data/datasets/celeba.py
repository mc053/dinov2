import os
import pickle
from .extended import ExtendedVisionDataset
from typing import Any, Callable, Optional, Tuple
from PIL import Image
from pathlib import Path

class CelebAOriginal(ExtendedVisionDataset):
    def __init__(self, root: str = os.path.dirname(os.path.abspath(__file__)), transforms=None, transform=None, target_transform=None, image_dir_name="CelebA_original"):
        super().__init__(root=root, transforms=transforms, transform=transform, target_transform=target_transform)
        self.root = Path(root).resolve()
        self.image_dir = self.root / "CelebA" / image_dir_name
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
        return 0  # TODO: Maybe return identity of person (?)

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

## class CelebAPixelated(CelebAOriginal) -> simply call super constructor with image_dir_name="CelebA_pixelated"
## class CelebAMasked(CelebAOriginal) ...
## class CelebABlurred(CelebAOriginal) ...
## class CelebADistorted(CelebAOriginal) ...