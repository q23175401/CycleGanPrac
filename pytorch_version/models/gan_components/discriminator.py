import torch
from torch import nn


class DiscrinimarotConvBlock(nn.Module):
    def __init__(self, in_depth, out_depth, stride=2, padding=1, ins_norm=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_depth, out_depth, 4, stride=stride, padding=padding, padding_mode="reflect", bias=True,),
            nn.InstanceNorm2d(out_depth) if ins_norm else nn.Identity(),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_depth=3, out_depths=[64, 128, 256, 512]):
        super().__init__()

        self.init_conv = DiscrinimarotConvBlock(in_depth, out_depths[0], ins_norm=False)

        previous_out_depth = out_depths[0]
        self.layer_stack = []
        for oi, out_depth in enumerate(out_depths[1:]):
            self.layer_stack.append(
                DiscrinimarotConvBlock(previous_out_depth, out_depth, stride=2 if oi != len(out_depths[1:]) - 1 else 1,)
            )
            previous_out_depth = out_depth
        self.layer_stack.append(
            nn.Conv2d(  # last layer to discriminate real or fake
                previous_out_depth, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect",
            )
        )
        self.layer_stack.append(nn.Sigmoid())

        self.entire_model = nn.Sequential(self.init_conv, *self.layer_stack)

    def forward(self, x):
        return self.entire_model(x)


def usage_test():
    sample_inputs = torch.randn([5, 3, 256, 256])
    dis_test = Discriminator(3)
    sample_outputs = dis_test(sample_inputs)
    print(sample_outputs.shape)


if __name__ == "__main__":
    usage_test()
