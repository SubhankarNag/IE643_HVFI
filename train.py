from utils.training_loop import train
from utils.config import * 

from choose_model import *

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


# deterministic runs
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True

model = Model()
if is_finetuning:
    model.flownet.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    # if the above does not work
    # model.load_model(pretrained_model_path, -1)

train(model)