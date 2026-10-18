"""Run a synthetic interface example with the included trained models."""
import argparse
from kaleid import Kaleid
import torch

parser = argparse.ArgumentParser()
parser.add_argument("system", choices=("nikon", "trepan", "olympus"), default="olympus", nargs="?")
args = parser.parse_args()
torch.set_num_threads(4)
model = Kaleid(args.system)
observation = model.demo_observation()
command = model.predict(observation)
print("Synthetic input: interface demonstration only.")
print("Normalized correction:", command)
print("Physical correction [mm, mm, mm, deg, deg] per body:")
print(model.to_physical(command).reshape(-1, 5))
