import numpy as np
import torch
import random

# -----------------------------------------------------------------------
# Only needed for multi gpu training
local_rank = 0 # used for distributed gpu, so use -1 or dont use at all
world_size = 4
# -----------------------------------------------------------------------


# hyperparameters
# NOTE: learning rate can be found in the training_loop.py file
max_epochs = 2 
batch_size = 6
num_workers = 0     # 4 
pin_memory = False  # True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_frames = 9          # use 9 when using context else 7
num_instances = 9       # max 42 or 33 instances/video incase of 7 or 9 frames respectively
is_context = True       # False if context is not used
is_finetuning = False   

# Logging paths
train_log_path = './new_train_log'
val_log_path = './new_val_log'
