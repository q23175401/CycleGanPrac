import albumentations as A  # for data augmentation
from albumentations.pytorch import ToTensorV2


class DataPreprocessor:
    def __init__(self) -> None:
        self.processes = A.Compose(
            [
                A.Resize(286, 286),
                A.RandomCrop(256, 256),
                # A.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2(),
            ]
        )

    def transform(self, data):
        return self.processes(image=data)["image"]
