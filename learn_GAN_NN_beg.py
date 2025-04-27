# ----------------------------------------------------------------------------
# Code by Łukasz Halbiniak
# GAN network CONV with FFT - GANFoxy2.0 - Version 2.0
# Code for generation new samples from previos database with samples
# All rights beong to Łukasz Halbiniak
# ----------------------------------------------------------------------------

# This file contains function to create nuerla network with learning algorthm

# Import
import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from NN_model import  Generator_CONV
from NN_model import  Discriminator_CONV

# Variables
ngpu = 0  # If 1 on PC is a GPU. If 0 GPU is no avalible
batch_size = 2
adversarial_loss = nn.BCELoss()
input_noise = 100
num_epochs = 500

#  Generator loss
def generator_loss(fake_output, label):
    gen_loss = adversarial_loss(fake_output, label)
    return gen_loss

# Discrimiator loss
def discriminator_loss(output, label):
    disc_loss = adversarial_loss(output, label)
    return disc_loss

#  Weights initialziation
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight,a=0.75,nonlinearity='leaky_relu')
    elif isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.kaiming_uniform_(m.weight,a=0.75,nonlinearity='leaky_relu')
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)



# Main function
def learn_GAN_NN_beg_main():
    print("")
    print("Mode: Learning neural network from beginning")
    print("Initialize PYTORCH")

    # Launching torch
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Loading data from file for learning
    print("")
    print("Loading learning data from 'prep_larning_data' folder")
    NN_data_final = np.load(
        os.path.dirname(os.path.realpath(__file__)) + "\\prep_learning_data\\" + "NN_data_FINAL" + ".npy")

    # Creating tensor for learning
    tensor_data = Variable(torch.Tensor(NN_data_final))
    train_loader = torch.utils.data.DataLoader(dataset=tensor_data, batch_size=batch_size, shuffle=False, num_workers=2)

    print("Data loaded")
    print("Creating Generator LSTM and Discriminator LSTM.")

    # Generator and discriminator lanuch
    generator_CONV = Generator_CONV().to(device)
    discriminator_CONV = Discriminator_CONV().to(device)

    # Printing data about generator and discriminator
    print("")
    print(generator_CONV)
    print("")
    print(discriminator_CONV)
    print("")

    print("Initialziation of weights")
    #generator_CONV.apply(weights_init)
    #discriminator_CONV.apply(weights_init)

    print("Creating optimilizer")

    # Optimizer initializing
    G_optimizer = optim.Adam(generator_CONV.parameters(), lr=0.0001)
    #D_optimizer = optim.Adam(discriminator_CONV.parameters(), lr=0.0001)
    D_optimizer = optim.RMSprop(discriminator_CONV.parameters(),lr=0.0000001)
    print("")
    print("Learning neural net")
    print("")

    # Main learning function
    for epoch in range(1, num_epochs + 1):

        #Information about actual loss
        D_loss_Backup = 0
        G_loss_Backup = 0

        for index, (real_samples) in enumerate(train_loader):
            D_optimizer.zero_grad()

            # Real samples in device
            real_samples = real_samples.to(device)

            # Creating targets (fake or real)
            flag_ones = torch.Tensor(np.random.uniform(low=0.99, high=1, size=(real_samples.size(0))))
            flag_zeros = torch.Tensor(np.random.uniform(low=0, high=0.01, size=(real_samples.size(0))))

            real_flag_samples = Variable(flag_ones.to(device))
            fake_flag_samples = Variable(flag_zeros.to(device))

            # Discriminator learn real data
            print("")
            print("1. DISC_REAL / ")

            output = discriminator_CONV(real_samples)
            D_real_loss = discriminator_loss(output, real_flag_samples.unsqueeze(1))
            D_real_loss.backward()

            # Discriminator learn fake data
            print("2. DISC_FAKE / ")
            noise_vector = (torch.randn(real_samples.size(0), input_noise, device=device))  # Batchsize, sequence, lenght, samples

            # Noise vector normalization to -1 and 1
            for q in range(real_samples.size(0)):
                min_n = torch.min(noise_vector[q, :])
                max_n = torch.max(noise_vector[q, :])
                if abs(min_n) > max_n:
                    noise_vector[q, :] = noise_vector[q, :] / (abs(min_n))

                else:
                    noise_vector[q, :] = noise_vector[q, :] / (abs(max_n))

            # Noise to device
            noise_vector = noise_vector.to(device)

            # Generating a fake sounds
            generated_fake_sound = generator_CONV(noise_vector)

            # Discriminator learning
            output = discriminator_CONV(generated_fake_sound.detach())
            D_fake_loss = discriminator_loss(output, fake_flag_samples.unsqueeze(1))
            D_fake_loss.backward()

            # D_loss for this neural network
            D_loss_Backup = D_real_loss + D_fake_loss

            # # Optimizer for discriminator
            print("3. DISC_OPTIM / ")
            D_optimizer.step()
            #
            # # Training generator
            # print("4. GEN_LEARN / ")
            # G_optimizer.zero_grad()
            # gen_output = discriminator_CONV(generated_fake_sound)
            # G_loss = generator_loss(gen_output, real_flag_samples.unsqueeze(1))
            # G_loss_Backup = G_loss
            # G_loss.backward()
            #
            # # Optimizer
            # print("5. GEN_OPTIM / ")
            # G_optimizer.step()

            # Printing stats
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\t'
                  % (epoch, num_epochs, index, len(train_loader),
                     D_loss_Backup, G_loss_Backup))

            f = open(dir_path + '\\dump_folder\\Results.txt', 'a')
            f.write('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\t'
                    % (epoch, num_epochs, index, len(train_loader),
                       D_loss_Backup, G_loss_Backup))
            f.write('\n')
            f.close()

        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': generator_CONV.state_dict(),
                'optimizer_state_dict': G_optimizer.state_dict(),
                'loss': D_loss_Backup,
            }, dir_path + "\\GAN_neural_net\\generator_CONV_state" + str(epoch) + ".bin")

            torch.save({
                'epoch': epoch,
                'model_state_dict': discriminator_CONV.state_dict(),
                'optimizer_state_dict': D_optimizer.state_dict(),
                'loss': G_loss_Backup,
            }, dir_path + "\\GAN_neural_net\\discriminator_CONV_state" + str(epoch) + ".bin")
