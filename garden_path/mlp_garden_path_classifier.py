from pathlib import Path
import sys
import os

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import text_utils
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset
torch.manual_seed(42)
np.random.seed(42)
from data_utils import *
from trajectory_analysis import *
import model_runs

# STATUS UPDATE: currently converging to 70% validation accuracy after 30 epochs

def set_device():
    if torch.cuda.is_available():
        device = "cuda"
        return device
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps" # this is for Apple Silicon
        return device
    else:
        device = "cpu"
        return device


device = torch.device(set_device())


def train_model(model, train_loader, test_loader, epochs, lr, name = 'Model'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"\nTraining {name} for {epochs} epochs...")
    start_time = time.time()
    final_val_acc = 0
    final_val_loss = 0
    accuracies = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for trajectories, labels in train_loader:
            trajectories, labels = trajectories.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(trajectories)
            print("outputs shape:", outputs.shape)
            print("labels shape:", labels.shape)
            print("labels dtype:", labels.dtype)
            print("labels sample:", labels[:2])
            loss = criterion(outputs, labels.squeeze())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        # EVAL

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for trajectories, labels in test_loader:
                trajectories, labels = trajectories.to(device), labels.to(device)
                outputs = model(trajectories)
                print("outputs shape:", outputs.shape)
                print("labels shape:", labels.shape)
                print("labels dtype:", labels.dtype)
                print("labels sample:", labels[:2])
                loss = criterion(outputs, labels.squeeze())
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        final_val_loss = val_loss / len(test_loader)
        final_val_acc = 100 * correct / total
        print(f"  Epoch [{epoch + 1}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {final_val_loss:.4f} | Val Acc: {final_val_acc:.2f}%")
        accuracies.append(final_val_acc)
    end_time = time.time()
    duration = end_time - start_time
    print(f"{name} - Final Accuracy: {final_val_acc:.2f}%, Time: {duration:.2f}s")
    return final_val_acc, final_val_loss, duration, accuracies


class MLP(nn.Module):
    def __init__(self, max_len, hidden_dim=32):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(max_len, hidden_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 32)
        self.fc4 = nn.Linear(32, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 16)
        self.fc7 = nn.Linear(16, 16)
        self.output = nn.Linear(16, 2)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        x = self.output(x)
        return x


data = pd.read_csv('Corpus/modified_stimulus.csv') # for command line
# data = pd.read_csv('../Corpus/modified_stimulus.csv') # for pyCharm
sentences = data['Stimulus']
labels = data['Ambiguity']
if os.path.exists("garden_path_trajectories.npy"):
    trajectories = np.load("garden_path_trajectories.npy", allow_pickle=True)
else:
    trajectories = []
    for i in range(len(sentences)):
        trajectories.append(model_runs.get_trajectory_cls(text_utils.incrementize(sentences[i])))
        print(i + 1)

# trajectories = []
# for i in range(len(sentences)):
#     print(i + 1)
#     trajectories.append(model_runs.get_trajectory_cls(text_utils.incrementize(sentences[i])))

np.save('negated_garden_path_trajectories.npy', np.array(trajectories, dtype=object), allow_pickle=True)
paired = [(traj, label) for traj, label in zip(trajectories, labels) if len(traj) >= 2]
trajectories_filtered, labels_filtered = list(zip(*paired))


x_train_raw, x_test_raw, y_train, y_test = train_test_split(
    list(trajectories_filtered), list(labels_filtered), test_size=0.2, random_state=42
)

# x_train = vectors_between(x_train_raw)  # list of [step0, step1], each shape (seq_len, 768)
# x_test = vectors_between(x_test_raw)

x_train = calc_distances(x_train_raw)
x_test = calc_distances(x_test_raw)


def get_max_len(x):
    return max((len(traj) for traj in x), default=0)


max_len = max(get_max_len(x_train), get_max_len(x_test))

print(f"x_train: {x_train}")

# Encode string labels -> integers
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)


# checking shapes
print(f"x_train[0] type: {type(x_train[0])}, len: {len(x_train[0])}")
print(f"x_train[0][0] shape: {np.array(x_train[0][0]).shape}")
print(f"len(x_train): {len(x_train)}, len(y_train): {len(y_train)}")
num_classes = len(le.classes_)


def to_tensor(x, desired_length):
    padded = []
    for traj in x:
        traj = np.array(traj, dtype=np.float32)  # shape: (seq_len,)
        pad_len = desired_length - len(traj)
        if pad_len > 0:
            traj = np.concatenate([traj, np.zeros(pad_len)])
        else:
            traj = traj[:desired_length]
        padded.append(traj)
    return torch.tensor(np.array(padded), dtype=torch.float32)  # (batch, max_len)



x_train_tensor = to_tensor(x_train, max_len)
x_test_tensor = to_tensor(x_test, max_len)

y_train_tensor = torch.tensor(y_train_enc, dtype=torch.long)
y_test_tensor = torch.tensor(y_test_enc, dtype=torch.long)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

acc, loss, t, accuracies = train_model(MLP(max_len), train_loader, val_loader, epochs=1500, name="MLP", lr=1e-4)

print(acc)
print(loss)
print(t)

plt.plot(accuracies)
plt.title("Accuracy Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()
