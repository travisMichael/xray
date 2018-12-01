import sys
import train


if len(sys.argv) < 3:
    print("Please enter the correct number of arguments.")
else:
    name_of_model = sys.argv[1]
    name_of_augmentation = sys.argv[2]
    train.train_model(name_of_model, name_of_augmentation, "../xray")
    print("The model has finished training. It has been saved to the output directory")
