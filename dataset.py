from glob import glob
from os import path
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import os


def pad_or_truncate(npz_f, target_length_0, target_length_1, selected_features=None):
    processed = {}
    for key in selected_features:
        if hasattr(npz_f, key):
            array = np.array(getattr(npz_f, key), dtype=np.float32)
            current_length_0 = array.shape[0]
            current_length_1 = array.shape[1]
            if current_length_0 == target_length_0 and current_length_1 == target_length_1:
                processed[key] = array
            elif current_length_0 > target_length_0:
                indices = [slice(None)] * array.ndim
                indices[0] = slice(0, target_length_0)
                if current_length_1 > target_length_1:
                    indices[1] = slice(0, target_length_1)
                    processed[key] = array[tuple(indices)]
                else:
                    # Pad axis 1 if needed after truncating axis 0
                    pad_width = [(0, 0)] * array.ndim
                    pad_width[0] = (0, 0)
                    pad_width[1] = (0, target_length_1 - current_length_1)
                    processed[key] = np.pad(array[tuple(indices)], pad_width, mode='constant', constant_values=0)
            else:
                padding_needed = target_length_0 - current_length_0
                pad_width = [(0, 0)] * array.ndim
                pad_width[0] = (0, padding_needed)
                if current_length_1 > target_length_1:
                    # Truncate axis 1 after padding axis 0
                    indices = [slice(None)] * array.ndim
                    indices[1] = slice(0, target_length_1)
                    processed[key] = np.pad(array, pad_width, mode='constant', constant_values=0)[tuple(indices)]
                else:
                    # Pad axis 1 as well
                    pad_width[1] = (0, target_length_1 - current_length_1)
                    processed[key] = np.pad(array, pad_width, mode='constant', constant_values=0)
    return processed




class MLPC2025(Dataset):

    base_dir = "MLPC2025_classification"

    def __init__(
            self,
            audio_features_dir="audio_features",
            labels_dir="labels",
            num_classes=54,
            features_length_0=250,
            selected_features=None
    ):
        self.features_length_0 = features_length_0
        self.audio_features_dir = path.join(self.base_dir, audio_features_dir)
        self.selected_features = selected_features if selected_features is not None else [
            'embeddings', 'melspectrogram', 'mfcc', 'mfcc_delta', 'mfcc_delta2',
            'flatness', 'centroid', 'flux', 'energy', 'power', 'bandwidth',
            'contrast', 'zerocrossingrate'
        ]
        self.labels_dir = path.join(self.base_dir, labels_dir)
        self.file_pairs = self._find_and_pair_files()
        self.num_classes = num_classes
        self.longest_feature = self.find_biggest_selected_features()
              

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
    
    def find_biggest_selected_features(self):
        filepath_features = self.file_pairs[0][0]
        features_npz = np.load(filepath_features)
        lengths = [getattr(features_npz.f, key).shape[1] for key in self.selected_features ]
        return max(lengths)

    def __getitem__(self, index):
        filepath_features, filepath_labels = self.file_pairs[index]
        
        features_npz = np.load(filepath_features)
        labels_npz = np.load(filepath_labels)
        features_dict = pad_or_truncate(features_npz.f, self.features_length_0, self.longest_feature, self.selected_features)
        labels_list =list(labels_npz.values())
        classes = list(labels_npz.keys())
        class_id = None
        for i, label in enumerate(labels_list):
            if label.max() > 0:
                class_id = i
                break
        class_name = classes[class_id]
        # Convert to PyTorch tensors (dict of tensors)
        features_tensor =  torch.from_numpy(np.array(list(features_dict.values())))

        
        return features_tensor, class_id, class_name

    def __len__(self):
        return len(self.file_pairs)

# data_loader = MLPC2025()
# for i in range(len(data_loader)):
#     features, class_id, class_name = data_loader[i]



