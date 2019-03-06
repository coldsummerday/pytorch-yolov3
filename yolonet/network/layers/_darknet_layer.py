from __future__ import division
import  torch
import torch.nn as nn
import torch.nn.functional as F


__all__=["MaxPoolStride1","Conv2dBatchLeaky","InterpolateUpsample",
         "EmptyLayer","RouteLayer",
         "ShortcutLayer"]


class Conv2dBatchLeaky(nn.Module):
    """ This convenience layer groups a 2D convolution, a batchnorm and a leaky ReLU.
    They are executed in a sequential manner.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Size of the kernel of the convolution
        stride (int or tuple): Stride of the convolution
        padding (int or tuple): padding of the convolution
        leaky_slope (number, optional): Controls the angle of the negative slope of the leaky ReLU; Default **0.1**
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,leaky_slope=0.1):
        super(Conv2dBatchLeaky, self).__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.leaky_slope = leaky_slope
        self.padding = padding
        # Layer
        self.layers = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False),
            nn.BatchNorm2d(self.out_channels), #, eps=1e-6, momentum=0.01),
            nn.LeakyReLU(self.leaky_slope, inplace=True)
        )

    def __repr__(self):
        s = '{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, negative_slope={leaky_slope})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, x):
        x = self.layers(x)
        return x

class YoloLayerInfo(object):
    def __init__(self,anchors,anchors_mask,reduction=32):
        self.anchors = anchors
        self.anchors_mask = anchors_mask
        self.reduction = reduction


class MaxPoolStride1(nn.Module):
    def __init__(self, kernel_size):
        super(MaxPoolStride1, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1

    def forward(self, x):
        padded_x = F.pad(x, (0, self.pad, 0, self.pad), mode="replicate")
        pooled_x = nn.MaxPool2d(self.kernel_size, self.pad)(padded_x)
        return pooled_x


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class RouteLayer(EmptyLayer):
    def __init__(self,index,route_layer_indexs):
        super(RouteLayer,self).__init__()
        self.route_layer_indexs = route_layer_indexs
        self.index = index


    def forward(self, output_cache):
        assert  type(output_cache)==dict
        if len(self.route_layer_indexs)==1:
            out = output_cache[self.route_layer_indexs[0] ]
        else:
            map_1 = output_cache[self.route_layer_indexs[0]]
            map_2 = output_cache[self.route_layer_indexs[1]]
            out = torch.cat((map_1,map_2),1)
        return out
    def __repr__(self):

        return f"route {self.route_layer_indexs}"




class ShortcutLayer(EmptyLayer):
    def __init__(self,index,from_):
        assert type(index)==int
        assert  type(from_)==int
        self.layer_index = index
        self.from_ = from_
        super(ShortcutLayer,self).__init__()
    def forward(self, output_cache):
        out = output_cache[self.layer_index-1] + output_cache[self.layer_index + self.from_]
        return out

    def __repr__(self):

        return f"res {self.layer_index + self.from_}"




class InterpolateUpsample(nn.Module):
    ##pytorch 中的upsample
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(InterpolateUpsample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input):
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)

    def extra_repr(self):
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.mode
        return info

class Upsample(nn.Module):
    def __init__(self, stride=2):
        super(Upsample, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert (x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        ws = stride
        hs = stride
        x = x.view(B, C, H, 1, W, 1).expand(B, C, H, stride, W, stride).contiguous().view(B, C, H * stride, W * stride)
        return x


#

class ReOrgLayer(nn.Module):
    def __init__(self, stride=2):
        super(ReOrgLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        assert (x.data.dim() == 4)
        B, C, H, W = x.data.shape
        hs = self.stride
        ws = self.stride
        assert (H % hs == 0), "The stride " + str(self.stride) + " is not a proper divisor of height " + str(H)
        assert (W % ws == 0), "The stride " + str(self.stride) + " is not a proper divisor of height " + str(W)
        x = x.view(B, C, H // hs, hs, W // ws, ws).transpose(-2, -3).contiguous()
        x = x.view(B, C, H // hs * W // ws, hs, ws)
        x = x.view(B, C, H // hs * W // ws, hs * ws).transpose(-1, -2).contiguous()
        x = x.view(B, C, ws * hs, H // ws, W // ws).transpose(1, 2).contiguous()
        x = x.view(B, C * ws * hs, H // ws, W // ws)
        return x
