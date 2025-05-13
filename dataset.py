from glob import glob
from os import path
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import os


def pad_or_truncate(npz, target_length_0):
    processed = {}
    for key, value in list(npz.items()):
        array = np.array(value, dtype=np.float32)
        current_length_0 = array.shape[0]
        # Only operate on dimension 0
        if current_length_0 == target_length_0:
            processed[key] = torch.tensor(value, dtype=torch.float32)
        elif current_length_0 > target_length_0:
            indices = [slice(None)] * array.ndim
            indices[0] = slice(0, target_length_0)
            processed[key] = torch.from_numpy(array[tuple(indices)])
        else:
            padding_needed = target_length_0 - current_length_0
            pad_width = [(0, 0)] * array.ndim
            pad_width[0] = (0, padding_needed)
            processed[key] = torch.from_numpy(np.pad(array, pad_width, mode='constant', constant_values=0))
    return processed




class MLPC2025(Dataset):

    base_dir = "MLPC2025_classification"

    def __init__(
        self,
        audio_features_dir="audio_features",
        labels_dir="labels",
        features_length_0=250,
    ):
        self.features_length_0 = features_length_0
        self.audio_features_dir = path.join(self.base_dir, audio_features_dir)
        self.labels_dir = path.join(self.base_dir, labels_dir)
        self.file_pairs = self._find_and_pair_files()

    def _find_and_pair_files(self):
        """Finds feature files and pairs them with existing corresponding label files."""
        feature_files = sorted(glob(path.join(self.audio_features_dir, "*.npz")))
        label_files = sorted(glob(path.join(self.labels_dir, "*.npz")))
        paired_files = []
        for feature_file in feature_files:
            # Extract the base name without extension
            base_name = os.path.splitext(os.path.basename(feature_file))[0]
            # Construct the corresponding label file path, adding "_labels"
            label_file = os.path.join(self.labels_dir, f"{base_name}_labels.npz")
            if label_file in label_files:
                paired_files.append((feature_file, label_file))
        print(f"Found {len(paired_files)} matching feature/label pairs.")
        
        return paired_files
    


    def __getitem__(self, index):
        filepath_features, filepath_labels = self.file_pairs[index]
        features_npz = np.load(filepath_features)
        labels_npz = np.load(filepath_labels)
        features_dict = pad_or_truncate(features_npz, self.features_length_0)
        labels_list =list(labels_npz.values())
        classes = list(labels_npz.keys())
        class_id = None
        for i, label in enumerate(labels_list):
            if label.max() > 0:
                class_id = i
                break
        class_name = classes[class_id]
        
        return {
            'melspectrogram': features_dict["melspectrogram"].unsqueeze(0), 
            'embeddings': features_dict["embeddings"],
            'mfcc': torch.from_numpy(np.array([features_dict["mfcc"], features_dict["mfcc_delta"], features_dict["mfcc_delta2"]])),
            'bandwidth': features_dict["bandwidth"].squeeze(),
            'centroid': features_dict["centroid"].squeeze(),
            'energy': features_dict["energy"].squeeze(),
            'flatness': features_dict["flatness"].squeeze(),
            'flux': features_dict["flux"].squeeze(),
            'power': features_dict["power"].squeeze(),
            'zerocrossingrate': features_dict["zerocrossingrate"].squeeze(),
            'class_id': class_id, 
            'class_name': class_name,
        }

    def __len__(self):
        return len(self.file_pairs)

# data_loader = MLPC2025()
# for i in range(len(data_loader)):
#     first = data_loader[i]



