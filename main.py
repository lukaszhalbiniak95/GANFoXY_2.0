# ----------------------------------------------------------------------------
# Code by Łukasz Halbiniak
# GAN network CONV with FFT - GANFoxy2.0 - Version 2.0
# Code for generation new samples from previos database with samples
# All rights beong to Łukasz Halbiniak
# ----------------------------------------------------------------------------

# Importing python files
import prep_training_data
import prep_data_load
import learn_GAN_NN_beg
import gen_new_sample
# Main function
def main():
    mode = 0  # Selectable variable
    exit_var = 1
    print("GANFoXY 2.0")
    print("Author: Lukasz Halbiniak")

    while exit_var == 1:
        print("")
        print("Please type what do you want to do (input variable 1, 2... n):")

        print("1. Prepare variations of samples from 'training_data' folder")
        print("2. Prepare material for NN from 'prepared_training_data' folder")
        print("3. Learning neural network from beginning")
        print("4. Learning neural network from previous model (relearning)")
        print("5. Generate new samples from defined model")
        print("6. Exit")
        mode = input()

        print("You choose mode: " + mode)
        if mode == "1":
            prep_training_data.prep_training_data_main()  # Preparing training samples to another folder
            print("Finish work")
        elif mode == "2":
            prep_data_load.prep_data_load_main()  # Preparing data for neural network
            print("Finish work")
        elif mode == "3":
            learn_GAN_NN_beg.learn_GAN_NN_beg_main() # Learning neural network from beginning
            print("Finish work")
        elif mode == "4":
            # learn_GAN_NN_cont.learn_GAN_NN_cont_main() # Continuing learning model
            print("Finish work")
        elif mode == "5":
            gen_new_sample.gen_new_sample_main() # Generating new samples
            print("Finish work")
        elif mode == "6":
            print("Exiting from program")
            exit_var =0
        else:
            print("Wrong input. Try again") # Error typo


# Definition of main fuction
if __name__ == "__main__":
    main()
