import sys
import train


train.train_model("cnn", "original", "../xray/data")

print(str(len(sys.argv)))

if len(sys.argv) < 2:
    print("This is the name of the script: " + sys.argv[0])
else:
    name_of_model = sys.argv[1]
    print("cnn" in name_of_model)
    print("Name of modal: " + sys.argv[1])
