# ----------------------------------------------------------------------------
# Code by Łukasz Halbiniak
# GAN network CONV with FFT - GANFoxy2.0 - Version 2.0
# Code for generation new samples from previos database with samples
# All rights beong to Łukasz Halbiniak
# ----------------------------------------------------------------------------

# This file contains function to create new sample from model

# Import
import os
from scipy.fft import irfft, rfft
from scipy.io import wavfile
import numpy as np
import torch
from NN_model import  Generator_CONV

# Variables
input_noise = 100
ngpu =0
sample_qty = 5
fixed_lenght_FFT = 64  # How many FFT
name_gen = "generator_CONV_state30.bin"

def gen_new_sample_main():
    print("")
    print("Mode: Generate new samples from model")

    # Loading PYTORCH
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    generator = Generator_CONV().to(device)

    # Loading model
    dir_path = os.path.dirname(os.path.realpath(__file__))
    Generator_load = torch.load(dir_path + "\\GAN_neural_net\\" + name_gen)
    generator.load_state_dict(Generator_load['model_state_dict'])

    # Generating a noise for model
    noise_vector = (torch.randn(sample_qty,input_noise, device=device))
    for q in range(sample_qty):
        min_n = torch.min(noise_vector[q, :])
        max_n = torch.max(noise_vector[q, :])

        if (abs(min_n) > max_n):
            noise_vector[q, :] = noise_vector[q, :] / (abs(min_n))

        else:
            noise_vector[q, :] = noise_vector[q, :] / (abs(max_n))

    noise_vector = noise_vector.to(device)

    # Generating samples
    print("Generating samples from neural network")
    torch.no_grad()
    generated_fake_samples = generator(noise_vector)
    generated_fake_samples = generated_fake_samples.detach()
    generated_fake_samples = generated_fake_samples.numpy()

    print("Creating samples to wav with IFFT")
    for q in range (sample_qty):
        data_IFFT_Real_Left = generated_fake_samples [q, 0, :]
        data_IFFT_Imag_Left = generated_fake_samples [q, 1, :]
        data_IFFT_Real_Right = generated_fake_samples [q, 2, :]
        data_IFFT_Imag_Right = generated_fake_samples [q, 3, :]
        finishef_IFFT_Left = []
        finishef_IFFT_Right = []

        # Doing IFFT
        for x in range(int(fixed_lenght_FFT)):
            IFFT_Real_Left = data_IFFT_Real_Left[x, :] * 4000000
            IFFT_Imag_Left = data_IFFT_Imag_Left[x, :] * 4000000
            IFFT_Real_Right = data_IFFT_Real_Right[x, :] * 4000000
            IFFT_Imag_Right = data_IFFT_Imag_Right[x, :] * 4000000

            IFFT_Left_channel = IFFT_Real_Left[:] + 1j * IFFT_Imag_Left[:]
            IFFT_Right_channel = IFFT_Real_Right[:] + 1j * IFFT_Imag_Right[:]

            IFFT_results_Left = irfft(IFFT_Left_channel)
            IFFT_results_Right = irfft(IFFT_Right_channel)

            if x == 0:
                finishef_IFFT_Left = IFFT_results_Left
                finishef_IFFT_Right = IFFT_results_Right
            else:
                finishef_IFFT_Left = np.hstack((finishef_IFFT_Left, IFFT_results_Left))
                finishef_IFFT_Right = np.hstack((finishef_IFFT_Right, IFFT_results_Right))

        # Saving one sample for test
        data_test_save = np.column_stack((finishef_IFFT_Left, finishef_IFFT_Right))
        wavfile.write(os.path.dirname(os.path.realpath(__file__)) + "\\dump_folder\\Test_sample" + str(q) +".wav", 44100,
                    data_test_save.astype(np.int16))