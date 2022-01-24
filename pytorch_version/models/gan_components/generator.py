import torch
from torch import nn


class GeneratorConvBlock(nn.Module):
    def __init__(self, in_depth, out_depth, is_down=True, use_activation=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_depth, out_depth, padding_mode="reflect", bias=True, **kwargs)
            if is_down
            else nn.ConvTranspose2d(in_depth, out_depth, **kwargs),
            nn.InstanceNorm2d(out_depth),
            nn.ReLU(inplace=True) if use_activation else nn.Identity(),
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_depth) -> None:
        super().__init__()
        self.block = nn.Sequential(
            GeneratorConvBlock(in_depth, in_depth, kernel_size=3, padding=1),
            GeneratorConvBlock(in_depth, in_depth, kernel_size=3, padding=1, use_activation=False),
        )

    def forward(self, x):
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    def __init__(self, in_depth=3, out_depth=3, n_feature=64, n_residuals=9) -> None:
        super().__init__()
        nf = n_feature  # alias
        self.init_conv = GeneratorConvBlock(in_depth, nf, is_down=True, kernel_size=7, stride=1, padding=3)
        self.down_blocks = nn.Sequential(
            GeneratorConvBlock(nf * 1, nf * 2, kernel_size=3, stride=2, padding=1),
            GeneratorConvBlock(nf * 2, nf * 4, kernel_size=3, stride=2, padding=1),
        )

        self.residual_blocks = nn.Sequential(*[ResidualBlock(nf * 4) for _ in range(n_residuals)])

        self.up_blocks = nn.Sequential(
            GeneratorConvBlock(nf * 4, nf * 2, is_down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
            GeneratorConvBlock(nf * 2, nf * 1, is_down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

        self.last_conv = nn.Sequential(
            nn.Conv2d(nf, out_depth, kernel_size=7, stride=1, padding=3, padding_mode="reflect"), nn.Tanh()
        )

        self.entire_model = nn.Sequential(
            self.init_conv, self.down_blocks, self.residual_blocks, self.up_blocks, self.last_conv
        )

    def forward(self, x):
        return self.entire_model(x)


class UnetBlock(nn.Module):
    """
    ​Unet最​外層
        down_conv
        leakyRelu
        ##submodule##
        up_conv
        insNorm
        Relu

    外層
        down_conv
        insNorm
        leakyRelu
        ##submodule##
        up_conv
        insNorm
        if dropOut(0.5)
        Relu

    最中心層
        down_conv
        Relu
        up_convins
        Norm
        Relu
    """

    def __init__(
        self, in_depth, inner_depth, out_depth, middle_unet_block=None, layerPos="innermost", use_dropout=False,
    ) -> None:
        super().__init__()
        self.layerPos = layerPos

        down_conv = nn.Conv2d(in_depth, inner_depth, kernel_size=4, stride=2, padding=1, padding_mode="reflect")

        if layerPos == "innermost":  # 最內層的Unet layer
            up_conv = nn.ConvTranspose2d(inner_depth, out_depth, kernel_size=4, stride=2, padding=1)
            model = [
                down_conv,
                nn.ReLU(True),
                up_conv,
                nn.InstanceNorm2d(out_depth),
                nn.ReLU(True),
            ]
        else:  # 中間各層的Unet Layer # 最外層的Unet Layer
            up_conv = nn.ConvTranspose2d(inner_depth * 2, out_depth, kernel_size=4, stride=2, padding=1)

            model = [
                down_conv,
                nn.Identity() if layerPos == "outermost" else nn.InstanceNorm2d(inner_depth),
                nn.LeakyReLU(0.2, True),
                middle_unet_block,  # 由於輸入會是 concat(model(x), x) in_depth*2
                up_conv,
                nn.Identity() if layerPos == "outermost" else nn.InstanceNorm2d(out_depth),
                nn.Identity() if not use_dropout else nn.Dropout(0.5),
                nn.Tanh() if layerPos == "outermost" else nn.ReLU(True),
            ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.layerPos == "outermost":
            return self.model(x)
        return torch.concat([x, self.model(x)], 1)  # concat on axis=1 (img depth)


class UnetGenerator(nn.Module):  # translate image to another image
    def __init__(self, in_depth=3, out_depths=3, n_feature=64, n_down=8):
        super().__init__()
        nf = n_feature  # alias

        # when num_down==7 image size [256, 256] at innermost is [1, 1]
        unet_block = UnetBlock(nf * 8, nf * 8, nf * 8, layerPos="innermost")

        for i in range(n_down - 5):
            unet_block = UnetBlock(nf * 8, nf * 8, nf * 8, unet_block, layerPos="middle")
        unet_block = UnetBlock(nf * 4, nf * 8, nf * 4, unet_block, layerPos="middle")
        unet_block = UnetBlock(nf * 2, nf * 4, nf * 2, unet_block, layerPos="middle")
        unet_block = UnetBlock(nf, nf * 2, nf, unet_block, layerPos="middle")

        unet_block = UnetBlock(in_depth, nf, out_depths, unet_block, layerPos="outermost")

        self.entire_model = unet_block

    def forward(self, x):
        return self.entire_model(x)


def usage_test():
    sample_inputs = torch.randn([5, 3, 256, 256])
    generator = ResnetGenerator()
    # generator = UnetGenerator()
    sample_outputs = generator(sample_inputs)
    print(sample_outputs.shape)  # must be equal to input shape


if __name__ == "__main__":
    usage_test()
