# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
from paddle.nn import (Conv2D, BatchNorm2D, Linear, Dropout, MaxPool2D,
                       AvgPool2D)
from paddle import ParamAttr
import paddle.nn.functional as F

from ..registry import BACKBONES
from ..weight_init import weight_init_
from ...utils import load_ckpt


class h_sigmoid(nn.Layer):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Layer):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Layer):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_t = nn.AdaptiveAvgPool3d((1, None, None))
        self.pool_h = nn.AdaptiveAvgPool3d((None, 1, None))
        self.pool_w = nn.AdaptiveAvgPool3d((None, None, 1))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)  # 减少通道数量
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_t = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, t, h, w = x.size()
        x_t = self.pool_t(x)
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # 变换Tensor的维度
        print(x_t,x_h,x_w)
        import sys
        sys.exit(0)

        y = paddle.cat([x_t, x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = paddle.split(y, [t,h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

class ConvBNLayer(nn.Layer):
    """Conv2D and BatchNorm2D layer.

    Args:
        in_channels (int): Number of channels for the input.
        out_channels (int): Number of channels for the output.
        kernel_size (int): Kernel size.
        stride (int): Stride in the Conv2D layer. Default: 1.
        groups (int): Groups in the Conv2D, Default: 1.
        act (str): Indicate activation after BatchNorm2D layer.
        name (str): the name of an instance of ConvBNLayer.

    Note: weight and bias initialization include initialize values and name the restored parameters, values initialization are explicit declared in the ```init_weights``` method.

    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        groups=1,
        act=None,
        name=None,
        data_format="NCHW",
    ):
        super(ConvBNLayer, self).__init__()
        self._conv = Conv2D(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=(kernel_size - 1) // 2,
                            groups=groups,
                            weight_attr=ParamAttr(name=name + "_weights"),
                            bias_attr=False,
                            data_format=data_format)
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]

        self._act = act

        self._batch_norm = BatchNorm2D(out_channels,
                                       weight_attr=ParamAttr(name=bn_name +
                                                             "_scale"),
                                       bias_attr=ParamAttr(bn_name + "_offset"),
                                       data_format=data_format)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        if self._act:
            y = getattr(paddle.nn.functional, self._act)(y)
        return y

class BottleneckBlock(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        shortcut=True,
        num_seg=8,
        name=None,
        data_format="NCHW",
    ):
        super(BottleneckBlock, self).__init__()
        self.data_format = data_format
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            act="relu",
            name=name + "_branch2a",
            data_format=data_format,
        )
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act="relu",
            name=name + "_branch2b",
            data_format=data_format,
        )

        self.conv2 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels * 4,
            kernel_size=1,
            act=None,
            name=name + "_branch2c",
            data_format=data_format,
        )
        self.ca = CoordAtt(inp=in_channels,oup=out_channels)

        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels * 4,
                kernel_size=1,
                stride=stride,
                name=name + "_branch1",
                data_format=data_format,
            )

        self.shortcut = shortcut
        self.num_seg = num_seg

    def forward(self, inputs):
        if self.data_format == "NCHW":
            shifts = F.temporal_shift(inputs, self.num_seg, 1.0 / self.num_seg)
        elif self.data_format == "NHWC":
            #N,H,W,C --> N,C,H,W
            inputs_trans = paddle.transpose(inputs, perm=[0, 3, 1, 2])
            shifts = F.temporal_shift(inputs_trans, self.num_seg,
                                      1.0 / self.num_seg)
            #N,C,H,W --> N,H,W,C
            shifts = paddle.transpose(shifts, perm=[0, 2, 3, 1])
        y = self.conv0(shifts)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = paddle.add(x=short, y=conv2)
        return F.relu(y)

class BasicBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 shortcut=True,
                 name=None,
                 data_format="NCHW"):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            filter_size=3,
            stride=stride,
            act="relu",
            name=name + "_branch2a",
            data_format=data_format,
        )
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            filter_size=3,
            act=None,
            name=name + "_branch2b",
            data_format=data_format,
        )

        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                filter_size=1,
                stride=stride,
                name=name + "_branch1",
                data_format=data_format,
            )

        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = paddle.add(short, conv1)
        y = F.relu(y)
        return y


@BACKBONES.register()
class ResNetTSM(nn.Layer):
    """ResNet TSM backbone.

    Args:
        depth (int): Depth of resnet model.
        pretrained (str): pretrained model. Default: None.
    """
    def __init__(self, depth, num_seg=8, data_format="NCHW", pretrained=None):
        super(ResNetTSM, self).__init__()
        self.pretrained = pretrained
        self.layers = depth
        self.num_seg = num_seg
        self.data_format = data_format

        supported_layers = [18, 34, 50, 101, 152]
        assert self.layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(
                supported_layers, self.layers)

        if self.layers == 18:
            depth = [2, 2, 2, 2]
        elif self.layers == 34 or self.layers == 50:
            depth = [3, 4, 6, 3]
        elif self.layers == 101:
            depth = [3, 4, 23, 3]
        elif self.layers == 152:
            depth = [3, 8, 36, 3]

        in_channels = 64
        out_channels = [64, 128, 256, 512]

        self.conv = ConvBNLayer(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            act="relu",
            name="conv1",
            data_format=self.data_format,
        )
        self.pool2D_max = MaxPool2D(
            kernel_size=3,
            stride=2,
            padding=1,
            data_format=self.data_format,
        )

        self.block_list = []
        if self.layers >= 50:
            for block in range(len(depth)):
                shortcut = False
                for i in range(depth[block]):
                    if self.layers in [101, 152] and block == 2:
                        if i == 0:
                            conv_name = "res" + str(block + 2) + "a"
                        else:
                            conv_name = "res" + str(block + 2) + "b" + str(i)
                    else:
                        conv_name = "res" + str(block + 2) + chr(97 + i)
                    bottleneck_block = self.add_sublayer(
                        conv_name,
                        BottleneckBlock(
                            in_channels=in_channels
                            if i == 0 else out_channels[block] * 4,
                            out_channels=out_channels[block],
                            stride=2 if i == 0 and block != 0 else 1,
                            num_seg=self.num_seg,
                            shortcut=shortcut,
                            name=conv_name,
                            data_format=self.data_format,
                        ))
                    in_channels = out_channels[block] * 4
                    self.block_list.append(bottleneck_block)
                    shortcut = True
        else:
            for block in range(len(depth)):
                shortcut = False
                for i in range(depth[block]):
                    conv_name = "res" + str(block + 2) + chr(97 + i)
                    basic_block = self.add_sublayer(
                        conv_name,
                        BasicBlock(
                            in_channels=in_channels[block]
                            if i == 0 else out_channels[block],
                            out_channels=out_channels[block],
                            stride=2 if i == 0 and block != 0 else 1,
                            shortcut=shortcut,
                            name=conv_name,
                            data_format=self.data_format,
                        ))
                    self.block_list.append(basic_block)
                    shortcut = True

    def init_weights(self):
        """Initiate the parameters.
        Note:
            1. when indicate pretrained loading path, will load it to initiate backbone.
            2. when not indicating pretrained loading path, will follow specific initialization initiate backbone. Always, Conv2D layer will be initiated by KaimingNormal function, and BatchNorm2d will be initiated by Constant function.
            Please refer to https://www.paddlepaddle.org.cn/documentation/docs/en/develop/api/paddle/nn/initializer/kaiming/KaimingNormal_en.html
        """
        #XXX: check bias!!! check pretrained!!!

        if isinstance(self.pretrained, str) and self.pretrained.strip() != "":
            load_ckpt(self, self.pretrained)
        elif self.pretrained is None or self.pretrained.strip() == "":
            for layer in self.sublayers():
                if isinstance(layer, nn.Conv2D):
                    #XXX: no bias
                    weight_init_(layer, 'KaimingNormal')
                elif isinstance(layer, nn.BatchNorm2D):
                    weight_init_(layer, 'Constant', value=1)

    def forward(self, inputs):
        """Define how the backbone is going to run.

        """
        #NOTE: (deprecated design) Already merge axis 0(batches) and axis 1(clips) before extracting feature phase,
        # please refer to paddlevideo/modeling/framework/recognizers/recognizer2d.py#L27
        #y = paddle.reshape(
        #    inputs, [-1, inputs.shape[2], inputs.shape[3], inputs.shape[4]])

        #NOTE: As paddlepaddle to_static method need a "pure" model to trim. It means from
        #  1. the phase of generating data[images, label] from dataloader
        #     to
        #  2. last layer of a model, always is FC layer
        y = self.conv(inputs)
        y = self.pool2D_max(y)
        for block in self.block_list:
            y = block(y)
        return y


if __name__ == "__main__":
    ca = CoordAtt(64, 64)
    input = paddle.randn((1, 64, 8, 112, 112))
    out = ca(input)


