"""
Created on Tue Feb  19 16:20:47 2019

@author: Sidi Wu

"""

from os import walk
import os
from glob import glob
from os import getcwd, chdir
import numpy as np
import numpy, sys
from PIL import Image
import cv2
from PIL import Image
import PIL

def list_files1(directory, extension):
    L=[]
    L.append([])
    L.append([])   
    for (dirpath, dirnames, filenames) in walk(directory):
        for f in filenames:
            if f.endswith(extension):
               L[0].append(dirpath)
               L[1].append(f)               
    
    return L
def SubDirPath (d):
    return [os.path.join(d,f) for f in os.listdir(d)],os.listdir(d)

dire='/home/ge56cur/nas/Projectrs_test/KITTI'
ex='jpg'
#imfile='image_02'
c=list_files1(dire, ex)
print(len(c[0]))

for i in range(len(c[0])):
     Flow=[]
     path = c[0][i]+'/'+c[1][i]
     name = c[1][i].split('.')
     filename = c[0][i].split('/')
     inputimage = cv2.imread(path)
     image_size=inputimage.shape[0] // 32
     for j in range(31):
         cut = j * image_size
         crop1 = inputimage[cut : cut + image_size,:]
         j=j+1
         cut = j * image_size
         crop2 = inputimage[cut : cut + image_size,:]
         prvs = cv2.cvtColor(crop1,cv2.COLOR_BGR2GRAY)
         hsv = np.zeros_like(crop1)
         hsv[...,1] = 255
         next = cv2.cvtColor(crop2,cv2.COLOR_BGR2GRAY)

         flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
         mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
         hsv[...,0] = ang*180/np.pi/2
         hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
         bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
         #bgr1 = cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
         Flow.append(bgr)
     flows_comb = np.vstack( np.asarray(i) for i in Flow )
     flows_comb = PIL.Image.fromarray(flows_comb)

     DIR_TO_SAVE = '/home/ge56cur/nas/Projectrs_test/KITTIopticalflow/'+filename[-1]
     if not os.path.exists(DIR_TO_SAVE):
        os.makedirs(DIR_TO_SAVE)
     print(DIR_TO_SAVE)
     flows_comb.save(DIR_TO_SAVE+'/'+str(name[0])+'_opticalflow.jpg' )


