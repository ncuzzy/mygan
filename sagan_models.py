import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ConditionalBatchNorm2d import ConditionalBatchNorm2d
import numpy as np

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out

class Generator(nn.Module):
    """Generator."""

    def __init__(self, n_class=115, z_dim=128, ngf=64):
        super(Generator, self).__init__()
        layer1 = []
        layer2 = []
        layer3 = []
        layer4 = []
        layer5 = []
        layer6 = []

        layer1.append(torch.nn.utils.spectral_norm(nn.Linear(z_dim, ngf*16*4*4)))
        # 1024x4x4

        layer2.append(torch.nn.utils.spectral_norm(nn.ConvTranspose2d(ngf*16, ngf*8, 4, 2, 1)))

        # layer2.append(ConditionalBatchNorm2d(ngf*8, n_class))
        # layer2.append(nn.ReLU())
        self.cbn = ConditionalBatchNorm2d(ngf*8, n_class)
        self.cbn_relu = nn.ReLU()
        # 512x8x8

        layer3.append(torch.nn.utils.spectral_norm(nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1)))
        layer3.append(nn.BatchNorm2d(ngf*4))
        layer3.append(nn.ReLU())
        # 256x16x16

        layer4.append(torch.nn.utils.spectral_norm(nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1)))
        layer4.append(nn.BatchNorm2d(ngf*2))
        layer4.append(nn.ReLU())
        # 128x32x32

        layer5.append(torch.nn.utils.spectral_norm(nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1)))
        layer5.append(nn.BatchNorm2d(ngf))
        layer5.append(nn.ReLU())
        # 64x64x64
        
        layer6.append(torch.nn.utils.spectral_norm(nn.ConvTranspose2d(ngf, 3, 4, 2, 1)))
        layer6.append(nn.Tanh())
        # 3x128x128

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)
        self.l4 = nn.Sequential(*layer4)
        self.l5 = nn.Sequential(*layer5)
        self.l6 = nn.Sequential(*layer6)

        self.attn1 = Self_Attn( 128, 'relu')
        # self.attn2 = Self_Attn( 64,  'relu')

    def forward(self, z, y):
        # z = z.view(z.size(0), z.size(1), 1, 1)
        out=self.l1(z).view(z.size(0), -1, 4, 4)
        out=self.l2(out)
        out=self.cbn_relu(self.cbn(out,y))
        out=self.l3(out)
        out=self.l4(out)
        out = self.attn1(out)
        out=self.l5(out)
        # out,p2 = self.attn2(out)
        out = self.l6(out)
        
        return out


class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, n_class=115, ndf=64):
        super(Discriminator, self).__init__()
        layer1 = []
        layer2 = []
        layer3 = []
        layer4 = []
        layer5 = []
        layer6 = []

        layer1.append(torch.nn.utils.spectral_norm(nn.Conv2d(3, ndf, 4, 2, 1)))
        layer1.append(nn.BatchNorm2d(ndf))
        layer1.append(nn.LeakyReLU(0.1))
        # 64x64x64

        layer2.append(torch.nn.utils.spectral_norm(nn.Conv2d(ndf, ndf*2, 4, 2, 1)))
        layer2.append(nn.BatchNorm2d(ndf*2))
        layer2.append(nn.LeakyReLU(0.1))
        # 128x32x32

        layer3.append(torch.nn.utils.spectral_norm(nn.Conv2d(ndf*2, ndf*4, 4, 2, 1)))
        layer3.append(nn.BatchNorm2d(ndf*4))
        layer3.append(nn.LeakyReLU(0.1))
        # 256x16x16

        layer4.append(torch.nn.utils.spectral_norm(nn.Conv2d(ndf*4, ndf*8, 4, 2, 1)))
        layer4.append(nn.BatchNorm2d(ndf*8))
        layer4.append(nn.LeakyReLU(0.1))
        # 512x8x8

        layer5.append(torch.nn.utils.spectral_norm(nn.Conv2d(ndf*8, ndf*16, 4, 2, 1)))
        layer5.append(nn.BatchNorm2d(ndf*16))
        layer5.append(nn.LeakyReLU(0.1))
        # 1024x4x4

        layer6.append(torch.nn.utils.spectral_norm(nn.Linear(ndf*16, 1)))
        

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)
        self.l4 = nn.Sequential(*layer4)
        self.l5 = nn.Sequential(*layer5)
        self.l6 = nn.Sequential(*layer6)
        if n_class > 0:
            self.l_y = nn.Embedding(n_class, ndf*16)

        self.attn = Self_Attn(128, 'relu')

    def forward(self, x, y):
        h = self.l1(x)
        h = self.l2(h)
        h = self.attn(h)
        h = self.l3(h)
        h = self.l4(h)
        h = self.l5(h)
        h = torch.sum(h,(2,3))
        output = self.l6(h)
        # 64x1
        if y is not None:
            w_y = self.l_y(y)
            t = w_y * h
            print(output)
            output += torch.sum(t, 1, True)
        return output
