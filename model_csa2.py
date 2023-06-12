# Copyright 2020 by Andrey Ignatov. All Rights Reserved.

import torch.nn as nn
import torch


class CSANET(nn.Module):

    def __init__(self, instance_norm=True, instance_norm_level_1=False):
        super(CSANET, self).__init__()

        self.conv1= ConvMultiBlock(4, 32, 5, instance_norm=True)
        
        self.out_att1 = att_module(input_channels=32, ratio=2, kernel_size=3)
        self.out_att2 = att_module(input_channels=32, ratio=2, kernel_size=3)
        self.conv2 = ConvMultiBlock(32, 64, 3,  instance_norm=True)
        #self.transpose_conv = nn.ConvTranspose2d(96, 128, kernel_size=3, padding =1, stride =1)
        self.conv3 = ConvMultiBlock(96, 256, 3, instance_norm=True)
        self.pix_shuff = nn.PixelShuffle(2)
        self.conv4 = ConvLayer(64, 3, 3, stride=1, instance_norm=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        #print('x shape: ',x.shape)
        conv1 = self.conv1(x)
        #print('conv1 shape: ',conv1.shape)
        att1 = self.out_att1(conv1)
        #print('att1 shape: ',att1.shape)
        z1 = att1 + conv1
        att2 = self.out_att2(z1)
        #print('att2 shape: ',att2.shape)
        z2 = z1 + att2
        conv2 = self.conv2(z2)
        z3 = torch.cat([conv2, conv1], 1)
        #print('z3 shape : ',z3.shape)
        
        #t_conv = self.transpose_conv(z3)
        #print('t_conv shape: ',t_conv.shape)
        conv3 = self.conv3(z3)
        #print('conv3 shape: ',conv3.shape)
        pix_shuff = self.pix_shuff(conv3)
        #print('pix_shuff shape: ',pix_shuff.shape)
        conv4 = self.conv4(pix_shuff)
        #print('conv4 shape: ',conv4.shape)
        enhanced = self.sigmoid(conv4)        
        return enhanced

class ConvMultiBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, instance_norm):

        super(ConvMultiBlock, self).__init__()
        
        self.conv_a = ConvLayer(in_channels, out_channels, kernel_size, stride=1, instance_norm=instance_norm)
        self.conv_b = ConvLayer(out_channels, out_channels, kernel_size, stride=1, instance_norm=instance_norm)
        
    def forward(self, x):

        out = self.conv_a(x)
        output_tensor = self.conv_b(out)
        return output_tensor


class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, relu=True, instance_norm=False):

        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2

        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        self.instance_norm = instance_norm
        self.instance = None
        self.relu = None

        if instance_norm:
            self.instance = nn.InstanceNorm2d(out_channels, affine=True)

        if relu:
            self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):

        out = self.reflection_pad(x)
        out = self.conv2d(out)

        if self.instance_norm:
            out = self.instance(out)

        if self.relu:
            out = self.relu(out)

        return out

class depthwise_conv(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(depthwise_conv, self).__init__()
        
        reflection_padding = 2 * (kernel_size//2)
        
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.dw_conv =  nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size, dilation=2, groups=in_channels),nn.ReLU())
        self.point_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        
        y = self.reflection_pad(x)
        #print('y_depthwise shape: ',y.shape)
        conv1 = self.dw_conv(y)
        #print('depthwise_conv shape: ',conv1.shape)
        conv2 = self.point_conv(conv1)
        #print('point_conv shape: ',conv2.shape)
        out = self.sigmoid(conv2)
        return out

class SpatialAttention(nn.Module):
    def __init__(self, in_channels, kernel_size):
        
        super(SpatialAttention, self).__init__()
        
        self.dw = depthwise_conv(in_channels, kernel_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        #print('x_sa2 shape: ',x.shape)
        z= self.dw(x)
        #print('z_sa2 shape: ',z.shape)
        z1= self.sigmoid(z)
        out = x * z1
        #print('out_sa2 shape: ',out.shape)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_channels, in_channels // ratio, kernel_size = 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_channels // ratio, in_channels, kernel_size= 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        #print('x_ca_avg shape: ',avg_out.shape)
        max_out = self.fc(self.max_pool(x))
        #print('x_ca_max shape: ',max_out.shape)
        out = self.sigmoid(avg_out * max_out)
        #print('ca_out1 shape: ',out.shape)
        out_ca = out * x
        #print('ca_out shape: ',out_ca.shape)
        return out_ca
    
class att_module(nn.Module):
    
    def __init__(self, input_channels, ratio, kernel_size, instance_norm=False):
        super(att_module, self).__init__()
        #print(input_channels)
        self.conv1 = ConvLayer(in_channels= input_channels, out_channels=input_channels*2, kernel_size=3, stride=1, relu=True)
        self.conv2 = ConvLayer(in_channels=input_channels*2, out_channels=input_channels*2, kernel_size=1, relu=True, stride =1)
        
        self.ca = ChannelAttention(input_channels*2, ratio)
        #self.sa = SpatialAttention(in_channels, kernel_size=5, dilation=2)
        self.sa = SpatialAttention(input_channels*2, kernel_size)
        self.conv3 = ConvLayer(input_channels*4, input_channels, kernel_size=1, stride= 1, relu=True)
    
    def forward(self, x):
       
       conv1 = self.conv1(x)
       #print('conv1_att shape: ',conv1.shape)
       conv2 = self.conv2(conv1)
       #print('conv2_att shape: ',conv2.shape)
              
       z1 = self.ca(conv2)
       #print('z1_att shape: ',z1.shape)
       z2 = self.sa(conv2)
       #print('z2_att shape: ',z2.shape)
       out = self.conv3(torch.cat([z1, z2], 1))
       #print('out_att shape: ',out.shape)
       return out
   
class IEM_module(nn.Module):
    def __init__(self, in_channels):
    
        super(IEM_module, self).__init__()
        
        reflection_padding = 3//2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, bias=False)
        self.act1 = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc1 = nn.Linear(in_channels, 16)
        self.act2 = nn.ReLU()
        #self.conv2 = nn.Conv2d(in_channels, in_channels/4, kernel_size=1, bias=False)
        self.fc2 = nn.Linear(16, 8)
        self.act3 = nn.ReLU()
        self.fc3 = nn.Linear (8, 16)
        self.act4 = nn.Sigmoid()
        
    def forward(self, x):
       #print('x_att shape: ',x.shape)
       ref_pad = self.reflection_pad(x)
       #print('ref_pad_att shape: ',ref_pad.shape)
       conv1 = self.act1(self.conv1(ref_pad))
       #print('conv1 shape: ',conv1.shape)
       
       
       avg_pool_out = self.avg_pool(x)
       #print('avg_pool shape: ',avg_pool_out.shape)
       z1 = self.act2(self.fc1(avg_pool_out))
       #print('fc1 shape: ',z1.shape)
       z2 = self.act3(self.fc2(z1))
       #print('fc2 shape: ',z2.shape)
       z3 = self.act4(self.fc3(z2))
       #print('fc3 shape: ',z3.shape)
       
       out1 = conv1 * z3
       #print('out1 shape: ',out1.shape)
       
       out = x * out1
       #print('out shape: ',out.shape)
       return out
