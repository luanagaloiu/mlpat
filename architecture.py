import torch
import torch.nn as nn
import torch.nn.functional as F




class MyCNN(nn.Module):
    def __init__(self, num_features: int, num_classes=int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_features, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # self.conv6 = nn.Sequential(
        #     nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )
     
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            #nn.Dropout(0.5),
            nn.Linear(256*7*24, 512*3), 
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            #nn.Dropout(0.5),
            nn.Linear(512*3, 2048), 
            nn.ReLU()
        )
        self.fc3 = nn.Linear(2048, num_classes)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # x = self.conv6(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        # x = self.softmax(x)
        return x
    

# model = MyCNN(
#     num_features = 5,  # Number of input channels
#     num_classes = 5  # Number of output classes
# )