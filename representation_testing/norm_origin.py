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
import model_runs
from trajectory_analysis import *


# STATUS UPDATE: 65% validation accuracy after epoch 25, 50% validation after epoch 30,
# not super sure if this is an overfitting issue or an instability issue.

# TODO: More sentences (garden path and regular) **make sure they are balanced**
# TODO: Figure out better 1D convolution filter size
# TODO: Format run sentences to be grammatically correct and use actual punctuation


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
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    print(f"\nTraining {name} for {epochs} epochs...")
    start_time = time.time()
    final_val_acc = 0
    final_val_loss = 0
    accuracies = []
    train_accuracies = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_total = 0
        train_correct = 0
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
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        train_loss = running_loss / len(train_loader)
        final_train_acc = 100 * train_correct / train_total
        train_accuracies.append(final_train_acc)
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
                # print("outputs shape:", outputs.shape)
                # print("labels shape:", labels.shape)
                # print("labels dtype:", labels.dtype)
                # print("labels sample:", labels[:2])
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
    return final_val_acc, final_val_loss, duration, accuracies, train_accuracies

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, input_len):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(2)

        self.flatten = nn.Flatten()

        # Compute flattened size: 3 halvings → input_len // 8
        flat_size = 128 * (input_len // 8)

        self.fc1 = nn.Linear(flat_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.3)
        self.final = nn.Linear(512, num_classes)

        # forward() stays the same

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))

        x = self.flatten(x)
        x = self.relu1(self.fc1(x))  # Reuse relu1
        x = self.drop1(x)
        x = self.final(x)
        return x


holdout_data = pd.read_csv('Corpus/run_sentences_different_classes_no_punctuation.csv')
data = pd.read_csv('Corpus/run_sentences_different_classes_no_punctuation.csv') # for command line
# data = pd.read_csv('../Corpus/modified_stimulus.csv') # for pyCharm
sentences = data['sentence']
labels = data['type']
if os.path.exists("different_class_run_trajectories_no_punctuation.npy"):
    trajectories = np.load("different_class_run_trajectories_no_punctuation.npy", allow_pickle=True)
else:
    trajectories = []
    for i in range(len(sentences)):
        trajectories.append(model_runs.get_trajectory_cls(text_utils.incrementize(sentences[i])))
        print(i + 1)

# trajectories = []
# for i in range(len(sentences)):
#     print(i + 1)
#     trajectories.append(model_runs.get_trajectory_cls(text_utils.incrementize(sentences[i])))

np.save('different_class_run_trajectories_no_punctuation.npy', np.array(trajectories, dtype=object), allow_pickle=True)
paired = [(traj, label) for traj, label in zip(trajectories, labels) if len(traj) >= 2]
trajectories_filtered, labels_filtered = list(zip(*paired))


x_train_raw, x_test_raw, y_train, y_test = train_test_split(
    list(trajectories_filtered), list(labels_filtered), test_size=0.2, random_state=42
)

x_train = mag_from_origin(x_train_raw)  # list of [step0, step1], each shape (seq_len, 768)
x_test = mag_from_origin(x_test_raw)


def get_max_len(x):
    lengths = []
    for i in range(len(x)):
        lengths.append(len(x[i]))
    return max(lengths, default=0)

max_len = max([get_max_len(x_train), get_max_len(x_test)])

print(f"x_train: {x_train}")

# Encode string labels -> integers
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

num_classes = len(le.classes_)
cnn = SimpleCNN(num_classes, input_len=max_len)


def to_tensor(x, desired_length):
    for i in range(len(x)):
        for j in range(desired_length - len(x[i])):
            x[i].append(0)
    return torch.tensor(np.array(x), dtype=torch.float32)


x_train_tensor = to_tensor(x_train, max_len).unsqueeze(1)
x_test_tensor = to_tensor(x_test, max_len).unsqueeze(1)

y_train_tensor = torch.tensor(y_train_enc, dtype=torch.long)
y_test_tensor = torch.tensor(y_test_enc, dtype=torch.long)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

acc, loss, t, accuracies, train_accuracies = train_model(cnn, train_loader, val_loader, epochs=200, name="CNN", lr=1e-5)

print(acc)
print(loss)
print(t)

plt.plot(accuracies, label = "validation accuracy")
plt.plot(train_accuracies, label = "training accuracy")
plt.title("Accuracy vs Epoch, L2 Norm From Origin, no punctuation")
plt.xlabel("Epochs")
plt.ylabel("% Accuracy")
plt.legend()
plt.savefig("garden_path/saved_figs/incremental_magnitude_conv.png")
plt.show()
