from .customdateset import *
from .flare_detect_dataset import *

def get_train_dataset(inp_dir, tar_dir):
  return TrainLoadDataset(inp_dir, tar_dir)
  
def get_val_test_dataset(inp_dir, tar_dir):
  return ValTestLoadDataset(inp_dir, tar_dir)
