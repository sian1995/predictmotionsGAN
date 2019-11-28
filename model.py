import torch
from ops import *
from torch.autograd import Variable
import os
import numpy as np

class G_background(nn.Module):
    def __init__(self):
        super(G_background, self).__init__()
        self.model = nn.Sequential(
                deconv3d_video(1024,1024), #[-1,1024,2,4,4]
                batchNorm5d(1024),
                relu(),
                deconv3d(1024,512),
                batchNorm5d(512),
                relu(),
                deconv3d(512,256),
                batchNorm5d(256),
                relu(),
                deconv3d(256,128),
                batchNorm5d(128),
                relu(),
                deconv3d(128,3), 
		nn.Tanh()
                )


    def forward(self,x):
        #print('G_background Input =', x.size())
        out = self.model(x)
        print('G_background Output =', out.size())
        return out

class G_video(nn.Module):
    def __init__(self):
        super(G_video, self).__init__()
        self.model = nn.Sequential(
                deconv3d_video(1024,1024), #[-1,1024,2,4,4]
                batchNorm5d(1024),
                relu(),
                deconv3d(1024,512),
                batchNorm5d(512),
                relu(),
                deconv3d(512,256),
                batchNorm5d(256),
                relu(),
                deconv3d(256,128),
                batchNorm5d(128),
                relu(),
                )
    def forward(self,x):
        #print('G_video input =', x.size())
        out = self.model(x)
        print('G_video output =', out.size())
        return out
        #return 0

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encode = G_encode()
        self.background = G_background()
        self.video = G_video()
        #self.flow_net = nn.Sequential(conv2d(3,64),nn.relu())
        self.gen_net = nn.Sequential(deconv3d(128,3), nn.Tanh())
        self.mask_net = nn.Sequential(deconv3d(128,1), nn.Sigmoid())
        print('done initialization')

    def forward(self,x,y):
        #print('Generator input = ',x.size())
        x = x.squeeze(2)
        encoded = self.encode(x,y)
        encoded = encoded.unsqueeze(2)
        video = self.video(encoded) #[-1, 128, 16, 32, 32], which will be used for generating the mask and the foreground
        #print('Video size = ', video.size())

        foreground = self.gen_net(video) #[-1,3,32,64,64]
        #print('Foreground size =', foreground.size())
        
        mask = self.mask_net(video) #[-1,1,32,64,64]
        #print('Mask size = ', mask.size())
        mask_repeated = mask.repeat(1,3,1,1,1) # repeat for each color channel. [-1, 3, 32, 64, 64]
        #print('Mask repeated size = ', mask_repeated.size())
        
        #x = encoded.view((-1,1024,4,4))
        background = self.background(encoded) # [-1,3,32,64,64]
        print('Background size = ', background.size())
        #background_frames = background.unsqueeze(2).repeat(1,1,32,1,1) # [-1,3,32,64,64]
        out = torch.mul(mask,foreground) + torch.mul(1-mask, background)
        print('Generator out = ', out.size())        
        return out,mask

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential( # [-1, 3, 32, 64, 64]
                conv3d(3, 128), #[-1, 64, 16, 32, 32]
                lrelu(0.2), 
                conv3d(128,256), #[-1, 126,8,16,16]
                #batchNorm5d(256, 1e-3), 
                lrelu(0.2),
                conv3d(256,512), #[-1,256,4,8,8]
                #batchNorm5d(512, 1e-3),
                lrelu(0.2),
                conv3d(512,1024), #[-1,512,2,4,4]
                #batchNorm5d(1024,1e-3),
                lrelu(0.2),
                conv3d(1024,2, (2,4,4), (1,1,1), (0,0,0)) #[-1,2,1,1,1] because (2,4,4) is the kernel size
                )
        self.mymodules = nn.ModuleList([nn.Sequential(nn.Linear(2,1), nn.Sigmoid())])
        
    def forward(self, x):
        out = self.model(x).squeeze()
        out = self.mymodules[0](out)
        return out

class G_encode(nn.Module):
    def __init__(self):
        super(G_encode, self).__init__()
        self.model = nn.Sequential(
                conv2d(192,256),
                batchNorm4d(256),
                relu(),
                conv2d(256,512),
                batchNorm4d(512),
                relu(),
                conv2d(512,1024),
                batchNorm4d(1024),
                relu(),
                )
        self.feature1 = nn.Sequential(
                conv2d(3,128),
                relu(),
                )
        self.feature2 = nn.Sequential(
                conv2d(3,64),
                relu(),
                )
    def forward(self,x,y):
        #print('G_encode Input =', x.size())
        x1 = self.feature1(x)
        x2 = self.feature2(y)
        x = torch.cat((x1,x2),1)
        out = self.model(x)
        print('G_encode Output =', out.size())
        return out


