import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from opacus import PrivacyEngine
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------
# CNN Model
# ----------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*5*5, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# ----------------------
# Data Loading
# ----------------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28,28)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("chest_xray/train", transform=transform)
client_datasets = random_split(dataset, [len(dataset)//3]*3)

# ----------------------
# Federated Learning
# ----------------------
def train_local(model, loader, epochs=1):
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    privacy_engine = PrivacyEngine()

    model, optimizer, loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
    )

    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    return model.state_dict()

def federated_avg(global_model, client_states):
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.stack(
            [client_states[i][key].float() for i in range(len(client_states))]
        ).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model

# ----------------------
# Training Loop
# ----------------------
global_model = CNN().to(device)

for round in range(3):
    client_states = []
    for client_data in client_datasets:
        loader = DataLoader(client_data, batch_size=32, shuffle=True)
        local_model = CNN().to(device)
        local_model.load_state_dict(global_model.state_dict())
        state = train_local(local_model, loader)
        client_states.append(state)

    global_model = federated_avg(global_model, client_states)
    print(f"Round {round+1} completed")
