# ----------------------------------------------------------------------------
# Code by Łukasz Halbiniak
# GAN network CONV with FFT - GANFoxy2.0 - Version 2.0
# Code for generation new samples from previos database with samples
# All rights beong to Łukasz Halbiniak
# ----------------------------------------------------------------------------

# This file contains function to create variations of sounds from training_data folder

# Import
import os
import torch
import torchaudio
import torchaudio.functional
import ffmpy
from pydub import AudioSegment

def prep_training_data_main():

    print("")
    print("Mode: Prepare variations of samples from 'training_data' folder")
    print("Searching a files from folder")

    # Direction
    dir_path = os.path.dirname(os.path.realpath(__file__)) + "\\training_data"

    # Deleting filed from folder
    print("Clearing folder NN_sample_prep")
    remove = []
    remove_dir = os.path.dirname(os.path.realpath(__file__)) + "\\NN_sample_prep"
    for path in os.listdir(remove_dir):
        if os.path.isfile(os.path.join(remove_dir, path)):
                remove.append(path)

    for f in remove:
        os.chmod(remove_dir+ "\\"+f, 0o777)
        os.remove(remove_dir+ "\\"+f)

    # Files names list
    files = []

    # Function to load filenames
    for path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, path)):
            if path.find("wav") != -1:  # Checking if this is a wav or mp3 file and adding to list
                files.append(path)
            if path.find("mp3") != -1:
                files.append(path)

    if len(files) == 0:
        print("No files found")

    else:
        # How many samples
        print("Found files: " + str(len(files)))

        # Loading samples
        for x in range(len(files)):
            print("")
            print("Sample number:" + str(x))
            print("Name of sample: " + files[x])

            # Creating a waveform
            waveform, sample_rate = torchaudio.load(dir_path + "\\" + files[x], channels_first=False)

            # Adding noise and save
            noise_vec = torch.randn(waveform.size(0),2)
            snr_dbs = torch.tensor([40])
            noise_sample = torchaudio.functional.add_noise(waveform, noise_vec, snr_dbs)
            torchaudio.save(
                os.path.dirname(os.path.realpath(__file__)) + "\\NN_sample_prep" + "\\SNoise1_" + str(x) + ".wav",
                noise_sample, sample_rate, format="wav", channels_first=False)

            noise_vec = torch.randn(waveform.size(0), 2)
            snr_dbs = torch.tensor([20])
            noise_sample = torchaudio.functional.add_noise(waveform, noise_vec, snr_dbs)
            torchaudio.save(
                os.path.dirname(os.path.realpath(__file__)) + "\\NN_sample_prep" + "\\SNoise2_" + str(x) + ".wav",
                noise_sample, sample_rate, format="wav", channels_first=False)

            # Phaser and save
            phaser_sample = torchaudio.functional.phaser(waveform, sample_rate)
            torchaudio.save(
                os.path.dirname(os.path.realpath(__file__)) + "\\NN_sample_prep" + "\\SPhaser_" + str(x) + ".wav",
                phaser_sample, sample_rate, format="wav", channels_first=False)

            # Tempo change
            dir_main = dir_path + "\\" + files[x]
            dir_1 = os.path.dirname(os.path.realpath(__file__)) + "\\NN_sample_prep" + "\\STempo1_" + str(x) + ".wav"
            dir_2 = os.path.dirname(os.path.realpath(__file__)) + "\\NN_sample_prep" + "\\STempo2_" + str(x) + ".wav"
            dir_3 = os.path.dirname(os.path.realpath(__file__)) + "\\NN_sample_prep" + "\\STempo3_" + str(x) + ".wav"
            dir_4 = os.path.dirname(os.path.realpath(__file__)) + "\\NN_sample_prep" + "\\STempo4_" + str(x) + ".wav"
            dir_5 = os.path.dirname(os.path.realpath(__file__)) + "\\NN_sample_prep" + "\\STempo5_" + str(x) + ".wav"

            ff = ffmpy.FFmpeg(inputs={dir_main: None}, outputs={dir_1: ["-filter:a", "atempo=0.7"]})
            ff.run()
            ff = ffmpy.FFmpeg(inputs={dir_main: None}, outputs={dir_2: ["-filter:a", "atempo=0.9"]})
            ff.run()
            ff = ffmpy.FFmpeg(inputs={dir_main: None}, outputs={dir_3: ["-filter:a", "atempo=1"]})
            ff.run()
            ff = ffmpy.FFmpeg(inputs={dir_main: None}, outputs={dir_4: ["-filter:a", "atempo=1.1"]})
            ff.run()
            ff = ffmpy.FFmpeg(inputs={dir_main: None}, outputs={dir_5: ["-filter:a", "atempo=1.3"]})
            ff.run()


            # Pitchshift and save
            sound = AudioSegment.from_file(dir_main, format="wav")
            octave_1 = 0.1  # For decreasing, octave can be -0.5, -2 etc.
            octave_2 = 0.5  # For decreasing, octave can be -0.5, -2 etc.
            octave_3 = -0.1  # For decreasing, octave can be -0.5, -2 etc.
            octave_4 = -0.5  # For decreasing, octave can be -0.5, -2 etc.

            sample_rate_1 = int(sound.frame_rate * (2.0 ** octave_1))
            sample_rate_2 = int(sound.frame_rate * (2.0 ** octave_2))
            sample_rate_3 = int(sound.frame_rate * (2.0 ** octave_3))
            sample_rate_4 = int(sound.frame_rate * (2.0 ** octave_4))

            pitch_1_sound = sound._spawn(sound.raw_data, overrides={'frame_rate': sample_rate_1})
            pitch_2_sound = sound._spawn(sound.raw_data, overrides={'frame_rate': sample_rate_2})
            pitch_3_sound = sound._spawn(sound.raw_data, overrides={'frame_rate': sample_rate_3})
            pitch_4_sound = sound._spawn(sound.raw_data, overrides={'frame_rate': sample_rate_4})

            pitch_1_sound = pitch_1_sound.set_frame_rate(44100)
            pitch_2_sound = pitch_2_sound.set_frame_rate(44100)
            pitch_3_sound = pitch_3_sound.set_frame_rate(44100)
            pitch_4_sound = pitch_4_sound.set_frame_rate(44100)

            dir_1 = os.path.dirname(os.path.realpath(__file__)) + "\\NN_sample_prep" + "\\SPitch1_" + str(x) + ".wav"
            dir_2 = os.path.dirname(os.path.realpath(__file__)) + "\\NN_sample_prep" + "\\SPitch2_" + str(x) + ".wav"
            dir_3 = os.path.dirname(os.path.realpath(__file__)) + "\\NN_sample_prep" + "\\SPitch3_" + str(x) + ".wav"
            dir_4 = os.path.dirname(os.path.realpath(__file__)) + "\\NN_sample_prep" + "\\SPitch4_" + str(x) + ".wav"

            pitch_1_sound.export(dir_1, format="wav")
            pitch_2_sound.export(dir_2, format="wav")
            pitch_3_sound.export(dir_3, format="wav")
            pitch_4_sound.export(dir_4, format="wav")


    print("")
    print("Closing function")

