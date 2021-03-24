import paddle
import paddle.nn as nn


class h_sigmoid(nn.Layer):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU()

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
        self.pool_t = nn.AdaptiveAvgPool3D((None, 1, 1))
        self.pool_h = nn.AdaptiveAvgPool3D((1, None, 1))
        self.pool_w = nn.AdaptiveAvgPool3D((1, 1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv3D(inp, mip, kernel_size=1, stride=1, padding=0)  # 减少通道数量
        self.bn1 = nn.BatchNorm3D(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv3D(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv3D(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_t = nn.Conv3D(mip, oup, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x

        n, c, t, h, w = x.shape
        x_t = self.pool_t(x)
        x_h = self.pool_h(x).transpose((0,1,3,2,4))
        x_w = self.pool_w(x).transpose((0,1,4,2,3))  # 变换Tensor的维度
        #print(x_t.shape,x_h.shape,x_w.shape)
        #import sys
        #sys.exit(0)

        y = paddle.concat([x_t, x_h, x_w], axis=2)
        #print(y.shape)
        #import sys
        #sys.exit(0)
        print(y)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_t,x_h, x_w = paddle.split(y, [t,h,w], axis=2)
        x_h = x_h.transpose((0,1,3,2,4))
        x_w = x_w.transpose((0,1,4,2,3))

        a_t = self.sigmoid(self.conv_t(x_t))
        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w))

        out = identity * a_w * a_h * a_t
        return out

if __name__ == "__main__":
    import numpy as np
    ca = CoordAtt(64, 64)
    input = np.random.randn(1, 64, 8, 112, 112)
    input = paddle.to_tensor(input,dtype='float32')
    #print(type(input))
    out = ca(input)
    print(out.shape)
