from .gan_components import Discriminator, UnetGenerator, ResnetGenerator
import itertools
from .utils.manage_dataset import ImgToImgDataset, DataPreprocessor
from .utils.training import ImageBuffer
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CycleGan:
    def __init__(self, learning_rate=0.0002, betas=(0.5, 0.999)) -> None:
        self.DY = Discriminator().to(DEVICE)    # discriminate domain Y image real or fake
        self.DX = Discriminator().to(DEVICE)    # discriminate domain X image real or fake
        self.GY = ResnetGenerator().to(DEVICE)  # generate domain Y image from domain X
        self.GX = ResnetGenerator().to(DEVICE)  # generate domain X image from domain Y

        # define optimizers
        d_params = itertools.chain(self.DX.parameters(), self.DY.parameters())
        g_params = itertools.chain(self.GX.parameters(), self.GY.parameters())
        self.optimD = torch.optim.Adam(d_params, learning_rate, betas=betas)
        self.optimG = torch.optim.Adam(g_params, learning_rate, betas=betas)

        # buffer for training discriminators to reduce oscillation
        self.GX_img_buffer = ImageBuffer(50)
        self.GY_img_buffer = ImageBuffer(50)

        # pytorch gradient scaler
        self.d_scaler = torch.cuda.amp.grad_scaler.GradScaler()
        self.g_scaler = torch.cuda.amp.grad_scaler.GradScaler()

    def train_on_batch(self, batch_data, c_ratio, i_ratio):
        # size of batch_data = [2, batchsize, 3, 256, 256]
        real_x_imgs, real_y_imgs = batch_data
        real_x_imgs = real_x_imgs.to(DEVICE)
        real_y_imgs = real_y_imgs.to(DEVICE)

        lossGAN = nn.MSELoss()
        l1_loss = nn.L1Loss()
        # with torch.cuda.amp.autocast():  # just auto cast float16 to float64
        new_fake_y_imgs = self.GY(real_x_imgs)
        new_fake_x_imgs = self.GX(real_y_imgs)
        new_back_to_y_imgs = self.GY(new_fake_x_imgs)
        new_back_to_x_imgs = self.GX(new_fake_y_imgs)
        new_identity_y_img = self.GY(real_y_imgs)
        new_identity_x_img = self.GX(real_x_imgs)

        # store generator results and get training image from buffer
        self.GX_img_buffer.addImages(new_fake_x_imgs.detach())
        self.GY_img_buffer.addImages(new_fake_y_imgs.detach())            
        old_fake_x_imgs = self.GX_img_buffer.sampleImages().to(DEVICE)
        old_fake_y_imgs = self.GY_img_buffer.sampleImages().to(DEVICE)

        # discriminate images
        dy_disc_real_y_imgs = self.DY(real_y_imgs)  # must close to one  for DY
        dy_disc_fake_y_imgs = self.DY(old_fake_y_imgs)  # must close to zero for DY
        dx_disc_real_x_imgs = self.DX(real_x_imgs)  # must close to one  for DX
        dx_disc_fake_x_imgs = self.DX(old_fake_x_imgs)  # must close to zero for DX

        dx_disc_new_fake_x_imgs = self.DX(new_fake_x_imgs)
        dx_disc_new_fake_y_imgs = self.DY(new_fake_y_imgs)

        #### generator part ####
        loss_gx = lossGAN(dx_disc_new_fake_x_imgs, torch.ones_like(dx_disc_new_fake_x_imgs))  # need to fool DX
        loss_gy = lossGAN(dx_disc_new_fake_y_imgs, torch.ones_like(dx_disc_new_fake_y_imgs))  # need to fool DY
        loss_gan = loss_gx + loss_gy

        loss_cycle_consistency = l1_loss(new_back_to_y_imgs, real_y_imgs) + l1_loss(new_back_to_x_imgs, real_x_imgs)
        loss_identity = (l1_loss(new_identity_x_img, real_x_imgs) + l1_loss(new_identity_y_img, real_y_imgs))/2
        loss_generator = loss_gan + c_ratio * loss_cycle_consistency + i_ratio * loss_identity

        self.optimG.zero_grad()
        self.g_scaler.scale(loss_generator).backward()
        self.g_scaler.step(self.optimG)
        # self.g_scaler.scale(loss_generator).backward(retain_graph=True)

        #### discriminator part ####
        loss_dy_real_y = lossGAN(dy_disc_real_y_imgs, torch.ones_like(dy_disc_real_y_imgs))
        loss_dy_fake_y = lossGAN(dy_disc_fake_y_imgs, torch.zeros_like(dy_disc_fake_y_imgs))
        loss_dy = loss_dy_real_y + loss_dy_fake_y

        loss_dx_real_x = lossGAN(dx_disc_real_x_imgs, torch.ones_like(dx_disc_real_x_imgs))
        loss_dx_fake_x = lossGAN(dx_disc_fake_x_imgs, torch.zeros_like(dx_disc_fake_x_imgs))
        loss_dx = loss_dx_real_x + loss_dx_fake_x

        loss_discriminator = (loss_dy + loss_dx) / 2

        self.optimD.zero_grad()
        self.d_scaler.scale(loss_discriminator).backward()
        self.d_scaler.step(self.optimD)

        self.d_scaler.update()
        self.g_scaler.update()

        loss_total = loss_generator + loss_discriminator
        return loss_total.cpu().detach().numpy()

    def learning_rate_decay(self, epoch_now, decay_after=100):
        assert epoch_now>=decay_after, 'epoch_now - decay_after can not be negtive'

        ratio = 1 - (epoch_now - decay_after + 1) / (decay_after + 1)

        old_learning_rate = self.optimD.param_groups[0]['lr']
        new_learning_rate = old_learning_rate * ratio
        for g in self.optimD.param_groups:
            g['lr'] = new_learning_rate

        for g in self.optimG.param_groups:
            g['lr'] = new_learning_rate


    def train(self, epochs, dataset: ImgToImgDataset, epoch_start_at=0, learning_rate_decay=100, batch_size=64, cycle_ratio=10, identity_ratio=10, ckpt_each=10):
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=6)
        
        for ep in range(epoch_start_at, epoch_start_at + epochs):
            progress_logger = tqdm(data_loader)
            total_batches = len(data_loader)
            error = 0

            # decay learning rate after some epochs
            if ep>=learning_rate_decay:
                self.learning_rate_decay(ep, learning_rate_decay)

            # get batch data from dataset and train on this batch
            for bi, batch_data in enumerate(progress_logger):
                error += self.train_on_batch(batch_data, cycle_ratio, identity_ratio)
                
            print(f'epoch: {ep+1} average error: {error/total_batches}')

            # save cyclegan after some epochs
            if ckpt_each>0 and (ep+1)%ckpt_each==0:
                self.save_checkpoint(f'checkpoint_{ep+1}_epoch')

    def discriminate(self, inputs, is_domain="X"):
        if is_domain == "X":
            return self.DX(inputs)
        else:
            return self.DY(inputs)

    def forward(self, inputs, to_domain="Y"):
        if to_domain == "Y":
            return self.GY(inputs)
        else:
            return self.GX(inputs)

    def save_checkpoint(self, checkpoint_name="checkpoint_xxx_epoch", save_directory="./pytorch_version/checkpoints/"):
        full_name = save_directory + checkpoint_name + ".pth.tar"

        # fmt: off
        checkpoint = {
                "DX": self.DX.state_dict(),
                "DY": self.DY.state_dict(),
                "GX": self.GX.state_dict(), 
                "GY": self.GY.state_dict(),
            "optimD": self.optimD.state_dict(),
            "optimG": self.optimG.state_dict(),
        }
        try:
            torch.save(checkpoint, full_name)
        except Exception as e:
            print(e)
            print('save model failed')

    def load_checkpoint(self, checkpoint_name="checkpoint_xxx_epoch", save_directory="./pytorch_version/checkpoints/"):
        full_name = save_directory + checkpoint_name + ".pth.tar"
        if os.path.exists(full_name):
            chackpoint = torch.load(full_name)
            self.DX.load_state_dict(chackpoint["DX"])
            self.DY.load_state_dict(chackpoint["DY"])
            self.GX.load_state_dict(chackpoint["GX"])
            self.GY.load_state_dict(chackpoint["GY"])
            self.optimD.load_state_dict(chackpoint["optimD"])
            self.optimG.load_state_dict(chackpoint["optimG"])

            #* then change learning_rate if you want
        else:
            print("Checkpoint does not exist. Load failed.")


def usage_test():
    cgan = CycleGan()
    dataset = ImgToImgDataset(DataPreprocessor())
    cgan.train(2, dataset, batch_size=4)


if __name__ == "__main__":
    usage_test()
