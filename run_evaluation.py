import argparse

from train.trainer import Trainer
from utils.base_utils import load_cfg

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str)
parser.add_argument('--model-path', type=str, default='model.pth')
flags = parser.parse_args()

trainer = Trainer(load_cfg(flags.cfg))
trainer.eval(flags.model_path)
