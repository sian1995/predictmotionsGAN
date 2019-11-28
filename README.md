# predictmotionsGAN
predict videos using GAN, see project_rs in the repository "paper samples"

1. model.py - the network structure of Generator and Discriminator to generate video from a single image 
2. opticalflowvideo.py - to generate optical flow of a image sequence 
3. data_loader1.py - to load image sequence together with the optical flow
4. train.py - using WGAN-GP loss to train the GAN
