import argparse

import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import os

from dcgan import Generator

def inference_dcgan_A(load_model_path, num_output):
    #'C:/Users/aiuser/Desktop/lai/DCGAN-PyTorch-master/exp_1/model/model_epoch_2700.pth'
    # Load the checkpoint file.
    state_dict = torch.load(load_model_path)

    # Set the device to run on: GPU or CPU.
    device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
    # Get the 'params' dictionary from the loaded state_dict.
    params = state_dict['params']

    # Create the generator network.
    netG = Generator(params).to(device)
    # Load the trained generator weights.
    netG.load_state_dict(state_dict['generator'])
    print(netG)

    print(load_model_path)
    # Get latent vector Z from unit normal distribution.
    noise = torch.randn(int(num_output), params['nz'], 1, 1, device=device)
    return netG, noise

def inference_dcgan_B(netG, noise, final_result_dir, model_name, repeat, index):
    # create final_result_dir/repeat/
    os.makedirs(final_result_dir+'/'+str(repeat), exist_ok=True)
    
    # Turn off gradient calculation to speed up the process.
    with torch.no_grad():
    # Get generated image from the noise vector using
    # the trained generator.
        generated_img = netG(noise).detach().cpu()
        # Display the generated image.
        plt.axis("off")
        # plt.title("Generated Images")
        # plt.imshow(np.transpose(vutils.make_grid(generated_img, padding=2, normalize=True), (1,2,0)))
        plt.imshow(np.transpose(vutils.make_grid(generated_img, padding=1, normalize=True)), interpolation='nearest')
        plt.savefig(final_result_dir+'/'+model_name+'/'+str(repeat)+'/'+str(index)+'.png',bbox_inches='tight',dpi=25,pad_inches=0.0)
        # fig.show()

#%%
# config
final_result_dir = 'C:/Users/aiuser/Desktop/lai/DCGAN-PyTorch-master/exp_1/inference_result/'
root_all_weight = 'C:/Users/aiuser/Desktop/lai/DCGAN-PyTorch-master/exp_1/model/'
load_model_path = os.listdir(root_all_weight)

repeat = 2
num_output = 500

for i in range(0, len(load_model_path)):
    tmp=[]
    countor_output = 0
    countor_repeat = 0
    while countor_repeat < repeat:
        while countor_output < num_output:
            netG, noise = inference_dcgan_A(root_all_weight+load_model_path[i], 1)
            noise_np = noise.cpu().numpy()
            # tensor[1, 100 ,1 ,1] --> array(100,)
            noise_np.resize(noise_np.shape[1],)
            # array 2 list
            list_noise_np = list(noise_np)
              
            if list_noise_np != tmp:
                inference_dcgan_B(netG, noise, final_result_dir, load_model_path[i], countor_repeat, countor_output)
                tmp += list_noise_np
                countor_output += 1
            else:
                continue
        countor_repeat += 1
