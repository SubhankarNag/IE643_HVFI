import torch
from torch.utils.data import DataLoader

from utils.dataset_frames import *
from utils.config import * 
from utils.training_loop import evaluate
from choose_model import *

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


model = Model()
model.flownet.load_state_dict(torch.load(pretrained_model_path, map_location=device))
# if the above does not work
# model.load_model(pretrained_model_path, -1)

# TRAIN
dataset_test = AccessMathDataset('train')
print("Train dataset size =",len(dataset_test))
test_data = DataLoader(dataset_test, batch_size=batch_size,
                        pin_memory=pin_memory, num_workers=num_workers)
evaluate(model, test_data, 0)


# VALIDATION
dataset_test = AccessMathDataset('validation')
print("Validation dataset size =",len(dataset_test))
test_data = DataLoader(dataset_test, batch_size=batch_size,
                        pin_memory=pin_memory, num_workers=num_workers)
evaluate(model, test_data, 0)


# TEST
dataset_test = AccessMathDataset('test')
print("Testing dataset size =",len(dataset_test))
test_data = DataLoader(dataset_test, batch_size=batch_size,
                        pin_memory=pin_memory, num_workers=num_workers)
evaluate(model, test_data, 0)