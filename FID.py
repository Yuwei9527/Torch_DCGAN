# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 16:55:02 2021

@author: yuwei
"""
# example of calculating the frechet inception distance in Keras for cifar10
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize
from keras.preprocessing.image import ImageDataGenerator
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
import tensorflow as tf
import os
from tqdm import tqdm
import statistics

config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# scale an array of images to a new size
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return asarray(images_list)

# calculate frechet inception distance
def calculate_fid(model, images1, images2):
	# calculate activations
	act1 = model.predict(images1)
	act2 = model.predict(images2)
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid

# prepare the inception v3 model
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))

#%%
# for one model
def torch_load_all_img(mode, root, repeat, batch_size):
    if mode ==-1:# orignal or base img
        real = dset.ImageFolder(root+'/',
                                transform=transforms.Compose([
                                    transforms.Resize((64, 64)),
                                    transforms.ToTensor(),
                                ]))
        
        dataloader = torch.utils.data.DataLoader(real, batch_size, shuffle=True)
        
        device = torch.device("cpu")
        
        for i, data in enumerate(dataloader, 0):
            # Transfer data tensor to GPU/CPU (device)
            real_data = data[0].to(device)
            real_data = real_data.reshape(real_data.shape[0], real_data.shape[2], real_data.shape[3], real_data.shape[1])
            
    elif mode == 0: #load one model
        real = dset.ImageFolder(root+'/'+str(repeat)+'/',
                                transform=transforms.Compose([
                                    transforms.Resize((64, 64)),
                                    transforms.ToTensor(),
                                ]))
        
        dataloader = torch.utils.data.DataLoader(real, batch_size, shuffle=True)
        
        device = torch.device("cpu")
        
        for i, data in enumerate(dataloader, 0):
            # Transfer data tensor to GPU/CPU (device)
            real_data = data[0].to(device)
            real_data = real_data.reshape(real_data.shape[0], real_data.shape[2], real_data.shape[3], real_data.shape[1])
            
    return real_data
#%%
# 真實瑕疵vs. 真實瑕疵
# root_A = 'C:/Users/aiuser/Desktop/lai/DCGAN-PyTorch-master/gan_compare_performance/sampling_Defect_A/'
# root_B = 'C:/Users/aiuser/Desktop/lai/DCGAN-PyTorch-master/gan_compare_performance/sampling_Defect_B/'
# A = os.listdir(root_A)

# sample_100_times = []
# all_model = []
# all_model_avg = []

# for i in tqdm(range(1, len(A)+1)):
#     real_data = torch_load_all_img(0, root_A, i, 8)
#     fake_data = torch_load_all_img(0, root_B, i, 8)

#     # print('Loaded', real_data.shape, fake_data.shape)
#     # convert integer to floating point values
#     # images1 = real_data.astype('float32')
#     # images2 = fake_data.astype('float32')
#     # resize images
#     images1 = scale_images(real_data.cpu(), (299,299,3))
#     images2 = scale_images(fake_data.cpu(), (299,299,3))
#     # print('Scaled', images1.shape, images2.shape)
#     # pre-process images
#     # images1 = preprocess_input(images1)
#     # images2 = preprocess_input(images2)
#     # calculate fid
#     # round
#     fid = round(calculate_fid(model, images1, images2), 3)
#     sample_100_times.append(fid)
    # print('FID: %.3f' % fid)

#%%
# 真實vs. 虛擬 -dcgan 
# work for muti_models
root_A = 'C:/Users/aiuser/Desktop/lai/DCGAN-PyTorch-master/gan_compare_performance/reality_defect/orange/'
root_B = 'C:/Users/aiuser/Desktop/lai/DCGAN-PyTorch-master/gan_compare_performance/sampling_DCGAN_do_process/'

all_model = []
all_model_avg = []

temp = os.listdir(root_B)# model_all_models
for j in tqdm(range(0, len(temp))):
    real_data = torch_load_all_img(-1, root_A, j, 8)# when mode==-1, j do nothing
    sample_100_times = []
    for i in range(1, 6):
        fake = dset.ImageFolder(root_B+'/'+temp[j]+'/'+str(i)+'/',
                                transform=transforms.Compose([
                                    transforms.Resize((64, 64)),
                                    transforms.ToTensor(),
                                ]))
        
        dataloader = torch.utils.data.DataLoader(fake, 8, shuffle=True)
        
        device = torch.device("cpu")
        
        for index, data in enumerate(dataloader, 0):
            # Transfer data tensor to GPU/CPU (device)
            fake_data = data[0].to(device)
            fake_data = fake_data.reshape(fake_data.shape[0], fake_data.shape[2], fake_data.shape[3], fake_data.shape[1])
    # resize images
        images1 = scale_images(real_data.cpu(), (299,299,3))
        images2 = scale_images(fake_data.cpu(), (299,299,3))
    # round
        fid = round(calculate_fid(model, images1, images2), 3)
        sample_100_times.append(fid)
        # print('FID: %.3f' % fid)
    all_model.append(sample_100_times)
    all_model_avg.append(statistics.mean(sample_100_times))
