from enum import Enum

class AnonymizationType(Enum):
    BLURRING = 0
    PIXELATION = 1
    DISTORTION = 2
    MASK_OUT = 3