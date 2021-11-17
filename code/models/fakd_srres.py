import models.common as common
import torch
import torch.nn as nn
import math
import torchsummary

def make_model(args, parent=False):
    return FakdSRRes(args)


class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(64, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(64, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        output = torch.add(output,identity_data)
        return output 

class FakdSRRes(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(FakdSRRes, self).__init__()
        self.args = args
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.LeakyReLU(0.2, inplace=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size=9, bias=False)]

        # define body module
        self.block_num = [int(n_resblocks/3), int(2*n_resblocks/3) - int(n_resblocks/3), n_resblocks - int(2*n_resblocks/3)]
        
        m_body1 = [common.ResBlock(conv, n_feats, kernel_size, bias=False, bn=True, act=act, res_scale=args.res_scale) for _ in range(self.block_num[0])]
        m_body2 = [common.ResBlock(conv, n_feats, kernel_size, bias=False, bn=True, act=act, res_scale=args.res_scale) for _ in range(self.block_num[1])]
        m_body3 = [common.ResBlock(conv, n_feats, kernel_size, bias=False, bn=True, act=act, res_scale=args.res_scale) for _ in range(self.block_num[2])]
        
        m_body3.append(conv(n_feats, n_feats, kernel_size))

        '''# define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size=9, bias=False)
        ]'''
        
        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            conv(n_feats, args.n_colors, kernel_size=9, bias=False)
        )
        
        self.head = nn.Sequential(*m_head)
        self.body1 = nn.Sequential(*m_body1)
        self.body2 = nn.Sequential(*m_body2)
        self.body3 = nn.Sequential(*m_body3)
        #self.tail = nn.Sequential(*m_tail)
        

    def forward(self, x):
        feature_maps = []
        x = self.sub_mean(x)
        x = self.relu(self.head(x))
        feature_maps.append(x)

        res = self.body1(x)
        feature_maps.append(res)
        res = self.body2(res)
        feature_maps.append(res)
        res = self.body3(res)
        feature_maps.append(res)
        
        res += x
        

        x = self.upscale4x(res)
        x = self.add_mean(x)

        return feature_maps, x
    
        
    
    def load_state_dict_teacher(self, state_dict):
        own_state = self.state_dict()
        tmp = [self.block_num[0], self.block_num[0] + self.block_num[1], self.block_num[0] + self.block_num[1] + self.block_num[2]]
        for name, param in state_dict.items():
            old_name = name
            if 'body' in name:
                a = name.split('.')
                m, n = 0, 0
                if int(a[1]) < tmp[0]:
                    m = 1
                    n = int(a[1])
                elif int(a[1]) < tmp[1]:
                    m = 2
                    n = int(a[1]) - tmp[0]
                elif int(a[1]) <= tmp[2]:
                    m = 3
                    n = int(a[1]) - tmp[1]
                a[0] = a[0] + str(m)
                a[1] = str(n)
                name = '.'.join(a)
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                own_state[name].copy_(param)
            else:
                print(old_name, name)


    def load_state_dict_student(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                own_state[name].copy_(param)
            else:
                print(name)  

class SRResWeight(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(SRRes, self).__init__()
        self.args = args
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.LeakyReLU(0.2, inplace=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
        #self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
        
        #elf.residual = self.make_layer(_Residual_Block, n_resblocks)

        #self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        

        #self.conv_output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4, bias=False)
        
        
        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size=9, bias=False)]

        # define body module
        self.block_num = [int(n_resblocks/3), int(2*n_resblocks/3) - int(n_resblocks/3), n_resblocks - int(2*n_resblocks/3)]
        
        m_body1 = [common.ResBlock(conv, n_feats, kernel_size, bias=False, bn=True, act=act, res_scale=args.res_scale) for _ in range(self.block_num[0])]
        m_body2 = [common.ResBlock(conv, n_feats, kernel_size, bias=False, bn=True, act=act, res_scale=args.res_scale) for _ in range(self.block_num[1])]
        m_body3 = [common.ResBlock(conv, n_feats, kernel_size, bias=False, bn=True, act=act, res_scale=args.res_scale) for _ in range(self.block_num[2])]
        
        m_body3.append(conv(n_feats, n_feats, kernel_size))

        '''# define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size=9, bias=False)
        ]'''
        
        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            conv(n_feats, args.n_colors, kernel_size=9, bias=False)
        )
        
        self.head = nn.Sequential(*m_head)
        self.body1 = nn.Sequential(*m_body1)
        self.body2 = nn.Sequential(*m_body2)
        self.body3 = nn.Sequential(*m_body3)
        #self.tail = nn.Sequential(*m_tail)
        

    def forward(self, x):
        
        x = self.sub_mean(x)
        x = self.relu(self.head(x))
        

        res = self.body1(x)
        
        res = self.body2(res)
        
        res = self.body3(res)
        
        res += x
        

        x = self.upscale4x(res)
        x = self.add_mean(x)

        return  x 


    def load_state_dict_teacher(self, state_dict):
        own_state = self.state_dict()
        tmp = [self.block_num[0], self.block_num[0] + self.block_num[1], self.block_num[0] + self.block_num[1] + self.block_num[2]]
        for name, param in state_dict.items():
            old_name = name
            if 'body' in name:
                a = name.split('.')
                m, n = 0, 0
                if int(a[1]) < tmp[0]:
                    m = 1
                    n = int(a[1])
                elif int(a[1]) < tmp[1]:
                    m = 2
                    n = int(a[1]) - tmp[0]
                elif int(a[1]) <= tmp[2]:
                    m = 3
                    n = int(a[1]) - tmp[1]
                a[0] = a[0] + str(m)
                a[1] = str(n)
                name = '.'.join(a)
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                own_state[name].copy_(param)
            else:
                print(old_name, name)


    def load_state_dict_student(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                own_state[name].copy_(param)
            else:
                print(name)        
                
