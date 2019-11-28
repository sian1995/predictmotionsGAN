'''
the code was adapted from https://github.com/batsa003/videogan 
'''
import torch
import torch.nn as nn
import numpy as np
import os
import sys
from tqdm import tqdm
from torchvision.transforms import ToPILImage
from PIL import Image
from torch.autograd import Variable
import time
import logging
from model import Discriminator
from model import Generator
from data_loader1 import DataLoader
import cv2 
import matplotlib.pyplot as plt 
#from logger import Logger
from utils import make_gif
import torchvision.transforms as transforms
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# Custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# Text Logger
def setup_logger(name):
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler('/home/ge56cur/nas/home/Projectrs_test/training_log_newtest0310_a.txt', mode='w',encoding='UTF-8')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger

# Saves [3, 64, 64] tensor x as image.
def save_img(x, filename):    
    x = x.squeeze()
    #x = denorm(x)
    to_pil = ToPILImage()
    img = to_pil(x)
    img.save(filename)

def to_variable(x, requires_grad = True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad)

def denorm(x):
    out = (x + 1.0) / 2.0
    m=nn.Tanh()
    return m(out)

def stretch(x):
    out = np.zeros_like(x)
    a = np.max(x)
    b = np.min(x)
    out = (x - b)/(b-a) * 255
    return out
    
def getopticallist(crop1,crop2):
    to_tensor = transforms.ToTensor()
    num = list(crop1.size())
    crop_size = 64
    f_out = torch.zeros((num[0], 3, crop_size, crop_size))
    for i in range(num[0]):
        crop11 = crop1[i,:,:,:,:]
        crop21 = crop2[i,:,:,:,:]
        crop11 = (crop11.squeeze()).cpu().detach().numpy()*255
        crop11 = crop11.astype('uint8')
        crop11 = np.transpose(crop11, (1,2,0))
        crop21 = (crop21.squeeze()).cpu().detach().numpy()*255
        crop21 = crop21.astype('uint8')
        crop21 = np.transpose(crop21, (1,2,0))
        #crop11 = crop11.cpu().detach().numpy().squeeze().transpose((1, 2, 0))
        #crop21 = crop21.cpu().detach().numpy().squeeze().transpose((1, 2, 0))
        prvs = cv2.cvtColor(crop11,cv2.COLOR_RGB2GRAY)
        hsv = np.zeros_like(crop11)
        hsv[...,1] = 255
        next = cv2.cvtColor(crop21,cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
        f_out[i,:,:,:] = to_tensor(bgr) 
    
    return f_out

num_epoch = 300
batchSize = 10
lr = 0.000002  #change at 15th January
l1_lambda = 10

text_logger = setup_logger('Train')
#logger = Logger('./logs')
    
discriminator = Discriminator()
generator = Generator()
discriminator.apply(weights_init)
generator.apply(weights_init)
#generator.load_state_dict(torch.load( '/home/ge56cur/nas/Projectrs_test/generator.pkl'))
#discriminator.load_state_dict(torch.load( '/home/ge56cur/nas/Projectrs_test//discriminator.pkl'))
if torch.cuda.is_available():
    discriminator.cuda()
    generator.cuda()

loss_function = nn.CrossEntropyLoss()
d_optim = torch.optim.Adam(discriminator.parameters(), lr, [0.5, 0.999])
g_optim = torch.optim.Adam(generator.parameters(), lr, [0.5, 0.999])

dataloader = DataLoader(batchSize)
data_size = len(dataloader.train_index)
num_batch = data_size//batchSize
print(data_size,'and',num_batch)
#print((dataloader.get_batch()).size())
#print(dataloader.train_index)
text_logger.info('Total number of videos for train = ' + str(data_size))
text_logger.info('Total number of batches per echo = ' + str(num_batch))

start_time = time.time()
counter = 0
DIR_TO_SAVE = "/home/ge56cur/nas/home/Projectrs_test/new_gen_videos_0310_a/"
if not os.path.exists(DIR_TO_SAVE):
    os.makedirs(DIR_TO_SAVE)
sample_input = None
sample_input_set = False


print('start training')
text_logger.info('start training')
for current_epoch in tqdm(range(1,num_epoch+1)):
    n_updates = 1
    print(current_epoch)
    #dataloader = DataLoader(batchSize)  #should initialize here!!
    for batch_index in range(num_batch):
        videos,flow = dataloader.get_batch() # [-1,32,3,64,64]
        videos=videos.permute(0,2,1,3,4) #[-1 3 32 64 64]
        print(flow.size())
        videos = to_variable(videos)
        flow = to_variable(flow)
        real_labels = to_variable(torch.LongTensor(np.ones(batchSize, dtype = int)), requires_grad = False)  #modify at 07. March
        fake_labels = to_variable(torch.LongTensor(np.zeros(batchSize, dtype = int)), requires_grad = False)
        if not sample_input_set:
            sample_input = videos[0:1,:,0:1,:,:]
            sample_input_set = True
        if n_updates % 2 == 1:
            discriminator.zero_grad()
            generator.zero_grad()
            outputs = discriminator(videos).squeeze() # [-1,2]
            fake_videos_d,mask = generator(videos[:,:,0:1,:,:],flow)
            outputs1 = discriminator(fake_videos_d).squeeze()
            d_loss = torch.mean(outputs1)-torch.mean(outputs)
            print('outputs =',outputs)
            print('outputs1 =',outputs1)
            # gradient penalty
            lam = 10
            alpha = torch.rand(batchSize,1)
            alpha = alpha.expand(batchSize, videos[:,:,0:1,:,:][0].nelement()).contiguous().view(batchSize, videos.size(1), videos[:,:,0:1,:,:].size(2), videos.size(3), videos.size(4))
            if torch.cuda.is_available():
               alpha = alpha.cuda()
            interpolates = alpha*videos.data+((1-alpha)*fake_videos_d.data)
            interpolates = Variable(interpolates, requires_grad=True)
            interpolates_label = discriminator(interpolates).squeeze()
            gradients = torch.autograd.grad(outputs=interpolates_label, inputs=interpolates,
            grad_outputs=torch.ones(interpolates_label.size()).cuda() if torch.cuda.is_available() else torch.ones(interpolates_label.size()),create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradients = gradients.view(gradients.size(0), -1)
            gradient_penalty = ((gradients.norm(2, dim=1)-1)**2).mean()
            d_loss = d_loss+lam*gradient_penalty
            #d_real_loss = loss_function(outputs, real_labels)
            #fake_videos_d,mask = generator(videos[:,:,0:1,:,:],flow)
            #outputs = discriminator(fake_videos_d).squeeze()
            #d_fake_loss = loss_function(outputs, fake_labels)
            #d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optim.step()
            info = {
                 'd_loss': d_loss.item()
            }
            for tag,value in info.items():
            #    logger.scalar_summary(tag, value, counter)
                 text_logger.info((str(tag)+','+str(value)+','+str(counter)))
            #     text_logger.info(tag,value,counter)
            del d_loss
        else:
            discriminator.zero_grad()
            generator.zero_grad()
            first_frame = videos[:,:,0:1,:,:]
            fake_videos,mask = generator(first_frame,flow)
            outputs = discriminator(fake_videos).squeeze()
            gen_first_frame = fake_videos[:,:,0:1,:,:]
            gen_second_frame = fake_videos[:,:,1:2,:,:]
            to_tensor = transforms.ToTensor()
            print(flow.size())
            gen_flow = getopticallist(gen_first_frame,gen_second_frame)
            print(gen_flow.size())
            #print('flow',torch.max(flow))
            #print('g_flow',torch.max(gen_flow))
            #print('frame',torch.max(first_frame))
            reg_loss_f = torch.abs(0.3-torch.mean(torch.abs(flow.cuda() - gen_flow.cuda()))) * l1_lambda/5*3 
            reg_loss = torch.mean(torch.abs(first_frame - gen_first_frame)) * l1_lambda/5*2
            
            #g_loss = loss_function(outputs, real_labels) + reg_loss + reg_loss_f
            g_loss = -torch.mean(outputs) + reg_loss + reg_loss_f
            g_loss.backward()
            g_optim.step()
            info = {
                'g_loss' : g_loss.item(),
            }
            del g_loss
            for tag,value in info.items():
            #    logger.scalar_summary(tag, value, counter)
                 text_logger.info((str(tag)+','+str(value)+','+str(counter)))
            #     text_logger.info(tag,value,counter)

            '''
            # Calculate validation loss
            videos = to_variable(dataloader.get_batch('test').permute(0,2,1,3,4)) # [64,3, 32, 64, 64]
            first_frame = videos[:,:,0:1,:,:]
            fake_videos = generator(first_frame)
            outputs = discriminator(fake_videos).squeeze()
            gen_first_frame = fake_videos[:,:,0:1,:,:]
            err = torch.mean(torch.abs(first_frame - gen_first_frame)) * l1_lambda
            g_val_loss = loss_function(outputs, real_labels) + err
            info = {
                'g_val_loss' : g_val_loss.data[0],
            }
            for tag,value in info.items():
                logger.scalar_summary(tag, value, counter)
            '''

        n_updates += 1


        counter+= 1

        if (batch_index + 1) % 10 == 0:

            save_img(flow[0].data.cpu(), DIR_TO_SAVE + 'original_flow_%s_%s_a.jpg' % (current_epoch, batch_index))
            save_img(gen_flow[0].data.cpu(), DIR_TO_SAVE + 'fake_flow_%s_%s_a.jpg' % (current_epoch, batch_index))
            make_gif(denorm(videos.data.cpu()[0]), DIR_TO_SAVE + 'original_gifs_%s_%s_b.gif' % (current_epoch, batch_index))
            save_img(denorm(first_frame[0].data.cpu()), DIR_TO_SAVE + 'fake_gifs_%s_%s_a.jpg' % (current_epoch, batch_index))
            make_gif(denorm(fake_videos.data.cpu()[0]), DIR_TO_SAVE + 'fake_gifs_%s_%s_b.gif' % (current_epoch, batch_index))
            make_gif(denorm(mask.data.cpu()[0]), DIR_TO_SAVE + 'mask__%s_%s_b.gif' % (current_epoch, batch_index))
            text_logger.info('Gifs saved at epoch: %d, batch_index: %d' % (current_epoch, batch_index))

        if current_epoch % 100 == 0:
            torch.save(generator.state_dict(), './generator1.pkl')
            torch.save(discriminator.state_dict(), './discriminator1.pkl')
            text_logger.info('Saved the model to generator1.pkl and discriminator1.pkl')
            
        # Decay the learning rate, modified!!!!
        if current_epoch % 1000 == 0:
            lr = lr / 10.0
            text_logger.info('Decayed learning rate to %.16f at epoch %d' % (lr,current_epoch))
            for param_group in d_optim.param_groups:
                param_group['lr'] = lr
            for param_group in g_optim.param_groups:
                param_group['lr'] = lr
        torch.cuda.empty_cache()
torch.save(generator.state_dict(), '/home/ge56cur/nas/home/Projectrs_test/generator1.pkl')
torch.save(discriminator.state_dict(), '/home/ge56cur/nas/home/Projectrs_test/discriminator1.pkl')
text_logger.info('Saved the model to generator1.pkl and discriminator1.pkl')
torch.cuda.empty_cache()

#print(videos.size())
#v=videos.unsqueeze(2).repeat(1,1,5,1,1,1)
#print(v.size())
text_logger.info('end training')
Training_log='/home/ge56cur/nas/home/Projectrs_test/training_log_newtest0310_a.txt'
print(videos.size())
with open(Training_log) as f:
    text=f.readlines()
d_loss=[]
g_loss=[]
count=0
for line in text:
    a=line.split(' ')
    c=a[7].split(',')
    if (c[0]=='d_loss'):
       print(float(c[1]))
       d_loss.append(float(c[1]))
       count=count+1
    if (c[0]=='g_loss'):
       g_loss.append(float(c[1]))
       count=count+1
#plt.figure(1)
#plt.subplot(211)
plt.plot(d_loss)
plt.ylabel('d_loss')
plt.savefig('/home/ge56cur/nas/home/Projectrs_test/new_d_loss_0310_a')

#plt.subplot(212)
plt.plot(g_loss)
plt.ylabel('g_loss')
#plt.show()
plt.savefig('/home/ge56cur/nas/home/Projectrs_test/new_g_loss_0310_a')
print('img show!')

