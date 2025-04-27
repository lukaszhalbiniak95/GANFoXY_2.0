# ----------------------------------------------------------------------------
# Code by Łukasz Halbiniak
# GAN network CONV with FFT - GANFoxy2.0 - Version 2.0
# Code for generation new samples from previos database with samples
# All rights beong to Łukasz Halbiniak
# ----------------------------------------------------------------------------

# This file contains function to create dataload for neural network

# Import
import os
from scipy.fft import irfft, rfft
from scipy.io import wavfile
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

#Variables
fixed_sample_rate = 44100  # [Hz]
fixed_sample_time = 2  # [seconds]
fixed_FFT = 1024  # FFT lenght
fixed_lenght_FFT = 85  # How many FFT
testing = 1  # Testing revers FFT
def prep_data_load_main():

    print("")
    print("Mode: Preparing data for neural network from folder NN_sample_prep")

    # Deleting filed from folder
    print("Clearing folder prep_learning_data")
    remove = []
    remove_dir = os.path.dirname(os.path.realpath(__file__)) + "\\prep_learning_data"
    for path in os.listdir(remove_dir):
        if os.path.isfile(os.path.join(remove_dir, path)):
            remove.append(path)

    for f in remove:
        os.chmod(remove_dir + "\\" + f, 0o777)
        os.remove(remove_dir + "\\" + f)

    print("Searching a files from folder")
    # Direction
    dir_path = os.path.dirname(os.path.realpath(__file__)) + "\\NN_sample_prep"

    # Files names
    files = []
    S_NN_Data = []
    # Function to load filenames
    for path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, path)):
            files.append(path)

    # Simple information if no samples
    if len(files) == 0:
        print("No files found")
    else:
        # How many samples
        print("Found files: " + str(len(files)))

        # Loading samples
        for x in range(len(files)):

            # Loading sample
            print("")
            print("Sample number:" + str(x) + " from:" + str(len(files)))
            samplerate, wav_point = wavfile.read(dir_path + "\\" + files[x])

            # Function for sample rate (must be fixed)
            print("Checking sample rate")
            if samplerate == fixed_sample_rate:
                print("Correct sample rate")
            else:
                print ("Not correct sample rate. Resampling")
                signal.resample(wav_point[0], fixed_sample_rate)
                signal.resample(wav_point[1], fixed_sample_rate)

            # Fixed sample lenght
            fixed_sample_length = fixed_sample_rate * fixed_sample_time  # How many samples with left and right channel

            print("Setting correct length of sample")
            # Checking sample_diff and making zeros array
            sample_diff = abs(fixed_sample_length - abs(len(wav_point)))
            zeros_diff = np.zeros(sample_diff)

            # Channel left and right with fixed length
            if fixed_sample_length > len(wav_point):
                left_channel = np.append(np.array(wav_point[:, 0]), zeros_diff)
                right_channel = np.append(np.array(wav_point[:, 1]), zeros_diff)
            else:
                left_channel = np.array(wav_point[0:fixed_sample_length, 0])
                right_channel = np.array(wav_point[0:fixed_sample_length, 1])


            #Creating FFT from samples
            print("FFT of sample: 1024 window")
            N = fixed_FFT
            S_Real_Left = []
            S_Imag_Left = []
            S_Real_Right = []
            S_Imag_Right = []
            for k in range(int(fixed_lenght_FFT)):
                x_data_Left = rfft(left_channel[N * k : N * k + N])
                x_data_Right = rfft(right_channel[N * k : N * k + N])

                x_FFT_Real_Left = x_data_Left.real
                x_FFT_Imag_Left = x_data_Left.imag
                x_FFT_Real_Right = x_data_Right.real
                x_FFT_Imag_Right = x_data_Right.imag

                S_Real_Left.append(x_FFT_Real_Left)
                S_Imag_Left.append(x_FFT_Imag_Left)
                S_Real_Right.append(x_FFT_Real_Right)
                S_Imag_Right.append(x_FFT_Imag_Right)

            S_Real_Left = np.array(S_Real_Left)
            S_Imag_Left = np.array(S_Imag_Left)
            S_Real_Right = np.array(S_Real_Right)
            S_Imag_Right = np.array(S_Imag_Right)

            # Normalziation
            S_Real_Left_Max = np.max(S_Real_Left)
            S_Real_Left_Min = np.min(S_Real_Left)

            S_Imag_Left_Max = np.max(S_Imag_Left)
            S_Imag_Left_Min = np.min(S_Imag_Left)

            S_Real_Right_Max = np.max(S_Real_Right)
            S_Real_Right_Min = np.min(S_Real_Right)

            S_Imag_Right_Max = np.max(S_Imag_Right)
            S_Imag_Right_Min = np.min(S_Imag_Right)

            # Normalization Real Left
            if S_Real_Left_Max > abs(S_Real_Left_Min):
                S_Real_Left = S_Real_Left / S_Real_Left_Max
            else:
                S_Real_Left = S_Real_Left / abs(S_Real_Left_Min)

            # Normalziation Imag Left
            if S_Imag_Left_Max > abs(S_Imag_Left_Min):
                S_Imag_Left = S_Imag_Left / S_Imag_Left_Max
            else:
                S_Imag_Left = S_Imag_Left / abs(S_Imag_Left_Min)

            # Normalziation Real Right
            if S_Real_Right_Max > abs(S_Real_Right_Min):
                S_Real_Right = S_Real_Right / S_Real_Right_Max
            else:
                S_Real_Right = S_Real_Right / abs(S_Real_Right_Min)

            # Normalziation Imag Right
            if S_Imag_Right_Max > abs(S_Imag_Right_Min):
                S_Imag_Right = S_Imag_Right / S_Imag_Right_Max
            else:
                S_Imag_Right = S_Imag_Right / abs(S_Imag_Right_Min)

            # Making matrix 4D
            if x == 0:
                S_NN_Data = np.stack((S_Real_Left,S_Imag_Left,S_Real_Right,S_Imag_Right),axis=0)
                S_NN_Data = np.expand_dims(S_NN_Data, axis=0)
            else:
                S_NN_Data_Buff = np.stack((S_Real_Left,S_Imag_Left,S_Real_Right,S_Imag_Right),axis=0)
                S_NN_Data = np.concatenate((S_NN_Data, np.expand_dims(S_NN_Data_Buff, axis=0)) , axis=0)

        if testing == 1:
            print("")
            print("Saving one sample in reverse to 'dump_folder'. Testing purpose")

            data_IFFT_Real_Left = S_NN_Data[1, 0, :]
            data_IFFT_Imag_Left = S_NN_Data[1, 1, :]
            data_IFFT_Real_Right = S_NN_Data[1, 2, :]
            data_IFFT_Imag_Right = S_NN_Data[1, 3, :]
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
            wavfile.write(os.path.dirname(os.path.realpath(__file__)) + "\\dump_folder\\Test_64sample.wav", 44100,
                          data_test_save.astype(np.int16))

        # Saving to flie
        print("")
        print("Saving to file")
        np.save(os.path.dirname(os.path.realpath(__file__)) + "\\prep_learning_data\\" + "NN_data_FINAL"+ ".npy", S_NN_Data)