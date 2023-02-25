from networks.decoders.resnet_dec import ResNet_D_Dec, ResNet_D_Dec_Color
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResShortCut_D_Dec(ResNet_D_Dec):

    def __init__(self, block, layers, out_channel=1, norm_layer=None, large_kernel=False, late_downsample=False, act_func='tanh'):
        super(ResShortCut_D_Dec, self).__init__(block, layers, out_channel, norm_layer, large_kernel, late_downsample=late_downsample)
        self.act_func = act_func
        print('activation function:', self.act_func)

    def forward(self, x, mid_fea, is_training=True):
        ret = {}
        fea1, fea2, fea3, fea4, fea5 = mid_fea['shortcut']
        x = self.layer1(x) + fea5 # 16x
        x = self.layer2(x) + fea4 # 8x
        x_os8 = self.refine_OS8(x)
        ret['feature_os8'] = x

        x = self.layer3(x) + fea3
        x_os4 = self.refine_OS4(x)

        x = self.layer4(x) + fea2
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x) + fea1
        x_os1 = self.refine_OS1(x)
       
        x_os4 = F.interpolate(x_os4, scale_factor=4.0, mode='bilinear', align_corners=False)
        x_os8 = F.interpolate(x_os8, scale_factor=8.0, mode='bilinear', align_corners=False)
        
        if self.act_func == 'tanh':
            x_os1 = (torch.tanh(x_os1) + 1.0) / 2.0
            x_os4 = (torch.tanh(x_os4) + 1.0) / 2.0
            x_os8 = (torch.tanh(x_os8) + 1.0) / 2.0
        elif self.act_func == 'softmax':
            x_os1 = torch.softmax(x_os1, dim=1)
            x_os4 = torch.softmax(x_os4, dim=1)
            x_os8 = torch.softmax(x_os8, dim=1)
        else:
            raise NotImplementedError

        ret['alpha_os1'] = x_os1
        ret['alpha_os4'] = x_os4
        ret['alpha_os8'] = x_os8
        ret['feature'] = x

        return ret

    def forward3(self, x, mid_fea, step=1, is_training=True):
        ret = {}

        if step == 1: 
            fea1, fea2, fea3, fea4, fea5 = mid_fea['shortcut']

            x = self.layer1(x) + fea5
            x = self.layer2(x) + fea4
            x_os8 = self.refine_OS8(x)

            x_os8 = (torch.tanh(x_os8) + 1.0) / 2.0

            ret['alpha_os8'] = x_os8
            ret['feature_os8'] = x

        elif step == 3:
            fea1, fea2, fea3 = mid_fea

            x_os8 = self.refine_OS8(x)

            x = self.layer3(x) + fea3
            x_os4 = self.refine_OS4(x)

            x = self.layer4(x) + fea2
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.leaky_relu(x) + fea1
            x_os1 = self.refine_OS1(x)
       
            x_os4 = F.interpolate(x_os4, scale_factor=4.0, mode='bilinear', align_corners=False)
            x_os8 = F.interpolate(x_os8, scale_factor=8.0, mode='bilinear', align_corners=False)
            
            x_os1 = (torch.tanh(x_os1) + 1.0) / 2.0
            x_os4 = (torch.tanh(x_os4) + 1.0) / 2.0
            x_os8 = (torch.tanh(x_os8) + 1.0) / 2.0

            ret['alpha_os1'] = x_os1
            ret['alpha_os4'] = x_os4
            ret['alpha_os8'] = x_os8
            ret['feature'] = x

        else:

            raise NotImplementedError

        return ret


class ResShortCut_D_Dec_Color(ResNet_D_Dec_Color):

    def __init__(self, block, layers, out_channel=1, norm_layer=None, large_kernel=False, late_downsample=False):
        super(ResShortCut_D_Dec_Color, self).__init__(block, layers, out_channel, norm_layer, large_kernel, late_downsample=late_downsample)

    def forward(self, x, mid_fea, is_training=True):
        ret = {}
        fea1, fea2, fea3, fea4, fea5 = mid_fea['shortcut']
        x = self.layer1(x) + fea5
        x = self.layer2(x) + fea4
        x_os8 = self.refine_OS8(x)
        c_os8 = self.color_OS8(x)
        ret['feature_os8'] = x

        x = self.layer3(x) + fea3
        x_os4 = self.refine_OS4(x)
        if is_training:
            c_os4 = self.color_OS4(x)

        x = self.layer4(x) + fea2
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x) + fea1
        x_os1 = self.refine_OS1(x)
        if is_training:
            c_os1 = self.color_OS1(x)
       
        x_os4 = F.interpolate(x_os4, scale_factor=4.0, mode='bilinear', align_corners=False)
        x_os8 = F.interpolate(x_os8, scale_factor=8.0, mode='bilinear', align_corners=False)

        if is_training:
            c_os4 = F.interpolate(c_os4, scale_factor=4.0, mode='bilinear', align_corners=False)
            c_os8 = F.interpolate(c_os8, scale_factor=8.0, mode='bilinear', align_corners=False)
        
        x_os1 = (torch.tanh(x_os1) + 1.0) / 2.0
        x_os4 = (torch.tanh(x_os4) + 1.0) / 2.0
        x_os8 = (torch.tanh(x_os8) + 1.0) / 2.0

        if is_training:
            c_os1 = (torch.tanh(c_os1) + 1.0) / 2.0
            c_os4 = (torch.tanh(c_os4) + 1.0) / 2.0
            c_os8 = (torch.tanh(c_os8) + 1.0) / 2.0

        ret['alpha_os1'] = x_os1
        ret['alpha_os4'] = x_os4
        ret['alpha_os8'] = x_os8

        if is_training:
            ret['color_os1'] = c_os1
            ret['color_os4'] = c_os4
            ret['color_os8'] = c_os8

        ret['feature'] = x

        return ret
