import os
from torch.utils.data import Dataset
from .data_preprocessor import DataPreprocessor
from PIL import Image
import numpy as np


class ImgToImgDataset(Dataset):
    def __init__(
        self,
        data_preprocessor: DataPreprocessor,
        x_domain_root="./pytorch_version/dataset/train/horse/",
        y_domain_root="./pytorch_version/dataset/train/zebra/",
    ):
        super().__init__()

        self.data_preprocessor = data_preprocessor
        self.x_domain_root = x_domain_root
        self.y_domain_root = y_domain_root

        self.x_filenames = os.listdir(self.x_domain_root)
        self.y_filenames = os.listdir(self.y_domain_root)

        self.x_len = len(self.x_filenames)
        self.y_len = len(self.y_filenames)
        if self.x_len >= self.y_len:
            self.dataset_len = self.x_len
        else:
            self.dataset_len = self.y_len

    def __len__(self):
        return self.dataset_len

    def get_images_array_by_id(self, index):  # get images without preprocessing
        x_filename = self.x_filenames[index % self.x_len]
        y_filename = self.y_filenames[index % self.y_len]

        x_path = self.x_domain_root + x_filename
        y_path = self.y_domain_root + y_filename

        x_image = np.array(Image.open(x_path).convert("RGB"))
        y_image = np.array(Image.open(y_path).convert("RGB"))
        return x_image, y_image

    def __getitem__(self, index):
        x_image, y_image = self.get_images_array_by_id(index)

        aug_x_img = self.data_preprocessor.transform(x_image)
        aug_y_img = self.data_preprocessor.transform(y_image)
        return (aug_x_img, aug_y_img)


def usage_test():
    dm = ImgToImgDataset(DataPreprocessor())
    print(dm[1][0].shape)
    print(dm[1][1].shape)


if __name__ == "__main__":
    usage_test()
