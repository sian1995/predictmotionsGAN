# encoding= utf-8  
'''
the code was adapted from https://github.com/batsa003/videogan 
'''
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
import os
import cv2 
import numpy as np
import logging
import scipy.misc
import torch.nn as nn
from torchvision.transforms import ToPILImage


KITTI_DATA_LISTING = '/home/ge56cur/videolistKITTI.txt'
#DATA_ROOT = '/home/ge56cur/lmf-nas/Übungsdaten/VPF_Wu/VondrickGolfDataset/frames-stable-many/'
# load images together with optical flow
class DataLoader(object):

    def __init__(self, batch_size = 5): 
        #reading data list
        self.batch_size = batch_size
        self.crop_size = 64
        self.frame_size = 32
        self.image_size = 128 
        self.train = None
        self.test = None
    
        # Shuffle video index.
        data_list_path = os.path.join(KITTI_DATA_LISTING_DATA_LISTING) #603776 video path
        with open(data_list_path, 'r',encoding='utf-8') as f:
            self.video_index = [x.strip() for x in f.readlines()]
            np.random.shuffle(self.video_index)
            #print(self.video_index)
        self.size = len(self.video_index)
        self.train_index = self.video_index[:self.size]
        #self.test_index = self.video_index[self.size//2:]

		# A pointer in the dataset
        self.cursor = 0
        logging.basicConfig(filename="logfilename.log", level=logging.INFO)

    def get_batch(self, type_dataset='train'):
        if type_dataset not in('train', 'test'):
            print('type_dataset = ', type_dataset,' is invalid. Returning None')
            return None

        dataset_index = self.train_index if type_dataset == 'train' else self.test_index
        if self.cursor + self.batch_size > len(dataset_index):
            self.cursor = 0
            #np.random.shuffle(dataset_index) #change at 15th January, turn the shuffle off
        
        t_out = torch.zeros((self.batch_size, self.frame_size, 3, self.crop_size, self.crop_size))
        to_tensor = transforms.ToTensor() # Transforms 0-255 numbers to 0 - 1.0.
        #img_out=np.zeros((self.batch_size, self.frame_size, self.crop_size, self.crop_size, 3))
        # return optical flow
        f_out = torch.zeros((self.batch_size, 3, self.crop_size, self.crop_size))
        for idx in range(self.batch_size):
            #video_path = os.path.join(DATA_ROOT, dataset_index[self.cursor])
            video_path = dataset_index[self.cursor]
            inputimage = cv2.imread(str(video_path))
            #print(str(video_path))
            #print('imageshape',inputimage.shape)
            # transfer BGR to RGB, 19.02
            inputimage= cv2.cvtColor(inputimage, cv2.COLOR_BGR2RGB)
            
            #count = inputimage.shape[0] / self.image_size
            count=32
            self.image_size=inputimage.shape[0] // count
            logging.info(video_path.encode('utf-8'))
            crop1 = inputimage[0 : self.image_size, :,:]
            #crop1 = cv2.resize(crop1,(crop1.shape[0],crop1.shape[0]),interpolation = cv2.INTER_AREA)
            #crop1 = cv2.resize(crop1, (self.crop_size, self.crop_size))
            crop2 = inputimage[self.image_size : 2*self.image_size, :,:]
            #crop2 = cv2.resize(crop2,(crop2.shape[0],crop1.shape[0]),interpolation = cv2.INTER_AREA)
            #crop2 = cv2.resize(crop2, (self.crop_size, self.crop_size))
            #print('crop1',crop1.shape)
            opticalf = getoptical(crop1,crop2)
            resf = cv2.resize(opticalf,(crop1.shape[0],crop1.shape[0]),interpolation = cv2.INTER_AREA)
            f_out[idx,:,:,:] = to_tensor(cv2.resize(resf, (self.crop_size, self.crop_size)))
            #f_out[idx,:,:,:] = to_tensor(resf) 
            for j in range(self.frame_size):
                if j < count:
                    cut = j * self.image_size
                else:
                    cut = (count - 1) * self.image_size
                if isinstance(cut,int):
                   middle = (inputimage.shape[1])//2
                   width = (self.image_size-1)//2
                   #crop = inputimage[cut : cut + self.image_size, middle-width:middle-width+self.image_size-1,:]
                   crop = inputimage[cut : cut + self.image_size, :,:]
                   crop = cv2.resize(crop,(crop1.shape[0],crop1.shape[0]),interpolation = cv2.INTER_AREA)
                   #area=(middle-width,0,middle-width+self.image_size-1,self.image_size)
                   temp_out = to_tensor(cv2.resize(crop, (self.crop_size, self.crop_size)))
                   #img1_out=cv2.resize(crop, (self.crop_size, self.crop_size))
                   temp_out = temp_out * 2 - 1
                t_out[idx,j,:,:,:] = temp_out
                #img_out[idx,j,:,:,:]=img1_out
#                for cc in range(3):
#                    temp_out[cc,:,:] -= temp_out[cc,:,:].mean() # (According to Line 123 in donkey_video2.lua)
                   
            
            
            self.cursor += 1
            #if self.cursor==1:
                 #print(idx,'and',j)
                 #print(inputimage.shape)
                 #print(crop)
                 #print((cv2.resize(crop, (self.crop_size, self.crop_size))).shape)
                 #print(temp_out)
                 #print(t_out[0,0,:,:,:])
                 #cv2.waitKey(0)
        return t_out,f_out

def getoptical(crop1,crop2):    
    prvs = cv2.cvtColor(crop1,cv2.COLOR_RGB2GRAY)
    hsv = np.zeros_like(crop1)
    hsv[...,1] = 255
    next = cv2.cvtColor(crop2,cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
    return bgr

def save_img(x, filename):    
    x = x.squeeze()
    #x = denorm(x)
    to_pil = ToPILImage()
    img = to_pil(x)
    img.save(filename)

def denorm(x):
    out = (x + 1.0) / 2.0
    m=nn.Tanh()
    return m(out)

d,f = DataLoader().get_batch('train')
print(list(d.size()))
d1 = d[:,0:1,:,:,:]
d2 = d[:,1:2,:,:,:]

#f_out = getopticallist(d1,d2)
#print(f_out.size())
#image1=(f_out[0,:,:,:])
image2=(f[1])
#print(f[0]-temp_out)
#save_img(d1,'compare2.jpg')
#save_img(image1,'testopticalflow3.jpg')
save_img(image2,'tesagain1.jpg')
#save_img(temp_out,'testopticalflow6.jpg')
#cv2.imwrite('testopticalflow7.png',bgr)
#print(image1)
#scipy.misc.imsave("/home/ge56cur/testopt.jpg",image1)
#print(f)
