# Code by Łukasz Halbiniak
# GAN network CONV with FFT - GANFoxy2.0 - Version 2.0
# Code for generation new samples from previos database with samples
# All rights beong to Łukasz Halbiniak
# ----------------------------------------------------------------------------

# This file contains NN model

# Import
import torch.nn as nn

# Variables
input_noise = 100

# Generator model
class Generator_CONV(nn.Module):
    def __init__(self):
        super(Generator_CONV, self).__init__()

        self.FC1 = nn.Linear(input_noise, 441)
        self.FC2 = nn.PReLU()

        self.FC3 = nn.Linear(441, 7056)
        self.FC4 = nn.PReLU()

        self.FC5 = nn.Linear(7056, 28224)
        self.FC6 = nn.PReLU() # 1 56 504

        self.CONV1 = nn.ConvTranspose2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0, bias=0)
        self.CONV2 = nn.BatchNorm2d(16)
        self.CONV3 = nn.PReLU() # 16 58 506

        self.CONV4 = nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0, bias=0)
        self.CONV5 = nn.BatchNorm2d(32)
        self.CONV6 = nn.PReLU() # 32 60 508

        self.CONV7 = nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0, bias=0)
        self.CONV8 = nn.BatchNorm2d(64)
        self.CONV9 = nn.PReLU() # 64 62 510

        self.CONV10 = nn.ConvTranspose2d(in_channels= 64, out_channels=4, kernel_size=3, stride=1, padding=0, bias=0)
        self.CONV11 = nn.Tanh()  # 1 56 504

    def forward(self, input):
        output = self.FC1(input)
        output = self.FC2(output)
        output = self.FC3(output)
        output = self.FC4(output)
        output = self.FC5(output)
        output = self.FC6(output)
        output = output.view(input.size(0), 1, 56, 504)
        output = self.CONV1(output)
        output = self.CONV2(output)
        output = self.CONV3(output)
        output = self.CONV4(output)
        output = self.CONV5(output)
        output = self.CONV6(output)
        output = self.CONV7(output)
        output = self.CONV8(output)
        output = self.CONV9(output)
        output = self.CONV10(output)

        return output

# Discriminator Model Class Definition
class Discriminator_CONV(nn.Module):
    def __init__(self):
        super(Discriminator_CONV, self).__init__()  # 4 64 512

        self.CONV1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, stride=1, padding=0, bias=1)
        self.CONV2 = nn.BatchNorm2d(64)
        self.CONV3 = nn.PReLU() # 64 62 510

        self.CONV4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0, bias=1)
        self.CONV5 = nn.BatchNorm2d(32)
        self.CONV6 = nn.PReLU() # 32 60 508

        self.CONV7 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=0, bias=1)
        self.CONV8 = nn.BatchNorm2d(1)
        self.CONV9 = nn.PReLU() # 1 58 506

        self.FLAT = nn.Flatten()

        self.FC1 = nn.Linear(29348, 7337)
        self.FC2 = nn.PReLU()

        self.FC3 = nn.Linear(7337, 1834)
        self.FC4 = nn.PReLU()

        self.FC5 = nn.Linear(1834, 1)
        self.FC6 = nn.Sigmoid()

    def forward(self, input):
        output = self.CONV1(input)
        output = self.CONV2(output)
        output = self.CONV3(output)
        output = self.CONV4(output)
        output = self.CONV5(output)
        output = self.CONV6(output)
        output = self.CONV7(output)
        output = self.CONV8(output)
        output = self.CONV9(output)
        output = self.FLAT(output)
        output = self.FC1(output)
        output = self.FC2(output)
        output = self.FC3(output)
        output = self.FC4(output)
        output = self.FC5(output)
        output = self.FC6(output)
        return output