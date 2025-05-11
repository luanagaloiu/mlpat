import torch
import argparse
import os
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from glob import glob
from os import path
from typing import Optional
import math
from dataset import ImagesDataset
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument("--submission", type=str)
args = parser.parse_args()

path_to_files = str(Path(args.submission).parent)
zip_file = f"{str(Path(args.submission).stem)}.zip"

original_wd = os.getcwd()
os.chdir(str(Path(args.submission).parent))

with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(".")

os.chdir(original_wd)

architecture = f"{path_to_files}/architecture.py"
trained_model = f"{path_to_files}/model.pth"

exec(open(architecture).read())
model.load_state_dict(torch.load(trained_model, map_location=torch.device('cpu')))
model.eval()

location_test_data = f"{os.getcwd()}/data/evaluation-scripts/1715726701_accuracy/test_data"
test_dataset = ImagesDataset(location_test_data, 100, 100, int)
test_dl = DataLoader(dataset=test_dataset, shuffle=False, batch_size=len(test_dataset))
correct = 0
total = 0

with torch.no_grad():
    for X, y, _, _ in test_dl:         
        y_pred = model(X)
        _, predicted = torch.max(y_pred.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

print(correct/total)
