import torch
from models import CycleGan
from models.utils.manage_dataset import DataPreprocessor, ImgToImgDataset
from PIL import Image
import numpy as np


def convert_tensor_to_image(gan_output):
    img = (gan_output + 1) * 255 / 2
    img_array = np.array(img.cpu().detach(), dtype=np.uint8).transpose((1, 2, 0))
    return Image.fromarray(img_array)


def get_imgs_tensor_from_dataset(dataset: ImgToImgDataset, start_i, length):
    tensor_x_imgs = []
    tensor_y_imgs = []
    for i in range(length):
        # get real image with type Image
        # real_img_x, real_img_y = dataset.get_images_array_by_id(start_i + i)
        # real_img_x = Image.fromarray(real_img_x)
        # real_img_y = Image.fromarray(real_img_y)

        # real_img_x.save(f"./pytorch_version/gan_result/{start_i+i}_real_x.png")
        # real_img_y.save(f"./pytorch_version/gan_result/{start_i+i}_real_y.png")

        # get processed image tensor
        test_x_imgs, test_y_imgs = dataset[start_i + i]
        tensor_x_imgs.append(np.array(test_x_imgs))
        tensor_y_imgs.append(np.array(test_y_imgs))
    tensor_x_imgs = torch.FloatTensor(np.array(tensor_x_imgs)).to("cuda")
    tensor_y_imgs = torch.FloatTensor(np.array(tensor_y_imgs)).to("cuda")
    return tensor_x_imgs, tensor_y_imgs


def test_result(cgan, dataset, start_i=0, test_length=30):
    # test on few imgs
    tensor_x_imgs, tensor_y_imgs = get_imgs_tensor_from_dataset(dataset, start_i, test_length)

    # generate fake images
    fake_y_imgs = cgan.forward(tensor_x_imgs, to_domain="Y")
    fake_x_imgs = cgan.forward(tensor_y_imgs, to_domain="X")

    ident_x_imgs = cgan.forward(tensor_x_imgs, to_domain="X")
    ident_y_imgs = cgan.forward(tensor_y_imgs, to_domain="Y")
    for i, fake_y_img in enumerate(fake_y_imgs):

        result_y_img = convert_tensor_to_image(fake_y_img)
        result_x_img = convert_tensor_to_image(fake_x_imgs[i])
        ident_x_img = convert_tensor_to_image(ident_x_imgs[i])
        ident_y_img = convert_tensor_to_image(ident_y_imgs[i])

        # save real and result image
        result_y_img.save(f"./pytorch_version/gan_result/{start_i+i}_xy.png")
        result_x_img.save(f"./pytorch_version/gan_result/{start_i+i}_yx.png")
        ident_x_img.save(f"./pytorch_version/gan_result/{start_i+i}_xx.png")
        ident_y_img.save(f"./pytorch_version/gan_result/{start_i+i}_yy.png")


def train(epoch=100, continue_train=100):
    cgan = CycleGan(learning_rate=1e-5)
    dataset = ImgToImgDataset(DataPreprocessor())

    if continue_train > 0:
        try:
            cgan.load_checkpoint(f"checkpoint_{continue_train}_epoch")
        except Exception:
            print("load faild")

    # train cycle gan
    cgan.train(
        epoch, dataset, epoch_start_at=continue_train, batch_size=1, cycle_ratio=10, identity_ratio=10, ckpt_each=20
    )

    # test result
    for i in range(30):
        test_result(cgan, dataset, start_i=i, test_length=1)


def load_test_result(epoch=100):
    cgan = CycleGan()
    preprocessor = DataPreprocessor()
    dataset = ImgToImgDataset(preprocessor)

    # test on checkpoints pretrained
    cgan.load_checkpoint(checkpoint_name=f"checkpoint_{epoch}_epoch")
    for i in range(30):
        test_result(cgan, dataset, start_i=i, test_length=1)


if __name__ == "__main__":
    # load_test_result(20)
    train(140, 60)
