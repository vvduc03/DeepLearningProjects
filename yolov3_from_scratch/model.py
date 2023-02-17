"""
Implementation of YOLOv3 architecture
"""

import torch
import torch.nn as nn

""" 
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)

        return x


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(
                2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1
            ),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )


class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

    def forward(self, x):
        outputs = []  # for each scale
        route_connections = []
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats,))

            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2),)
                    in_channels = in_channels * 3

        return layers

class YOLOV3_Byhand(nn.Module):
    def __init__(self, in_channels=3, num_classes=20):
        super().__init__()

        self.Block1 = nn.Sequential(CNNBlock(in_channels, 32, kernel_size=3, stride=1, padding=1),
                                    CNNBlock(32, 64, kernel_size=3, stride=2, padding=1))

        self.Block2 = nn.Sequential(ResidualBlock(64),
                                    CNNBlock(64, 128, kernel_size=3, stride=2, padding=1))

        self.Block3 = nn.Sequential(ResidualBlock(128),
                                    ResidualBlock(128),
                                    CNNBlock(128, 256, kernel_size=3, stride=2, padding=1))

        self.Block4_ = nn.Sequential(ResidualBlock(256),
                                     ResidualBlock(256),
                                     ResidualBlock(256),
                                     ResidualBlock(256),
                                     ResidualBlock(256),
                                     ResidualBlock(256),)

        self.Block5_ = nn.Sequential(CNNBlock(256, 512, kernel_size=3, stride=2, padding=1),
                                     ResidualBlock(512),
                                     ResidualBlock(512),
                                     ResidualBlock(512),
                                     ResidualBlock(512),
                                     ResidualBlock(512),
                                     ResidualBlock(512),)

        self.Block6_ = nn.Sequential(CNNBlock(512, 1024, kernel_size=3, stride=2, padding=1),
                                     ResidualBlock(1024),
                                     ResidualBlock(1024),
                                     ResidualBlock(1024),
                                     ResidualBlock(1024),
                                     CNNBlock(1024, 512, kernel_size=1),
                                     CNNBlock(512, 1024, kernel_size=3, padding=1))

        self.Block7_ = nn.Sequential(CNNBlock(1024, 256, kernel_size=1),
                                     nn.Upsample(scale_factor=2.0))

        self.Block8_ = nn.Sequential(CNNBlock(768, 256, kernel_size=1),
                                     CNNBlock(256, 512, kernel_size=3, padding=1))

        self.Block9_ = nn.Sequential(CNNBlock(512, 128, kernel_size=1),
                                     nn.Upsample(scale_factor=2.0))

        self.Block10_ = nn.Sequential(CNNBlock(384, 128, kernel_size=1),
                                     CNNBlock(128, 256, kernel_size=3, padding=1))


        self.ScalePredictBig = nn.Sequential(ResidualBlock(1024, use_residual=False),
                                             CNNBlock(1024, 512, kernel_size=1),
                                             ScalePrediction(512, num_classes)
                                             )

        self.ScalePredictMedium = nn.Sequential(ResidualBlock(512, use_residual=False),
                                                CNNBlock(512, 256, kernel_size=1),
                                                ScalePrediction(256, num_classes)
                                                )

        self.ScalePredictSmall = nn.Sequential(ResidualBlock(256, use_residual=False),
                                               CNNBlock(256, 128, kernel_size=1),
                                               ScalePrediction(128, num_classes)
                                               )

    def forward(self, x):
        outputs = []
        x = self.Block1(x)
        x = self.Block2(x)
        x = self.Block3(x)
        route3 = self.Block4_(x)
        x = self.Block4_(x)
        route2 = self.Block5_(x)
        x = self.Block5_(x)
        route1 = self.Block6_(x)
        x = self.Block6_(x)
        detec_big = self.ScalePredictBig(x)
        outputs.append(detec_big)

        route1 = self.Block7_(route1)
        route1 = torch.cat([route1, route2], dim=1)
        x = self.Block8_(route1)
        route1 = self.Block8_(route1)
        detec_medium = self.ScalePredictMedium(route1)
        outputs.append(detec_medium)

        x = self.Block9_(x)
        x = torch.cat([x, route3], dim=1)
        x = self.Block10_(x)
        detec_small = self.ScalePredictSmall(x)
        outputs.append(detec_small)

        return outputs


if __name__ == "__main__":
    num_classes = 20
    IMAGE_SIZE = 416
    model = YOLOV3_Byhand(num_classes=num_classes)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = model(x)
    assert model(x)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    assert model(x)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    assert model(x)[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    print("Success!")