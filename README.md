
Install Steps:
    conda create --name cnn
    conda activate cnn


Run Pre-process


The models have been saved already. To retrain the models, run the
following commands from the project's root directory.

Run Train
    python code/train_model.py cnn original
    python code/train_model.py cnn salt_and_pepper
    python code/train_model.py cnn rotation
    python code/train_model.py cnn reflection
    python code/train_model.py densenet original
    python code/train_model.py densenet salt_and_pepper
    python code/train_model.py densenet rotation
    python code/train_model.py densenet reflection


Run Test
    python code/loader/loader_test.py