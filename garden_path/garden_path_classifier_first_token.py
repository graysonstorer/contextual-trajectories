from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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
# torch.manual_seed(42) # uncomment for reproducibility
# np.random.seed(42) # uncomment for reproducibility
from data_utils import *
import model_runs
from trajectory_analysis import *
import text_utils
import os


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


def eval_model(model, hold_out_loader, name = 'Model'):
    model = model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for trajectories, labels in hold_out_loader:
            trajectories, labels = trajectories.to(device), labels.to(device)
            outputs = model(trajectories)
            # print("outputs shape:", outputs.shape)
            # print("labels shape:", labels.shape)
            # print("labels dtype:", labels.dtype)
            # print("labels sample:", labels[:2])
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    final_holdout_acc = 100 * correct / total
    return final_holdout_acc



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
            # print("outputs shape:", outputs.shape)
            # print("labels shape:", labels.shape)
            # print("labels dtype:", labels.dtype)
            # print("labels sample:", labels[:2])
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
    # 1D convolution network
    def __init__(self, seq_len, num_classes):
        super(SimpleCNN, self).__init__()

        # Block 1: 1 -> 32 filters
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=5, padding=1)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)

        # Block 2: 32 -> 64 filters
        self.conv2 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)

        # Block 3: 64 -> 128 filters (Added for complexity)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        with torch.no_grad():
            dummy = torch.zeros(1, 768, seq_len)
            dummy = self.pool1(torch.relu(self.bn1(self.conv1(dummy))))
            dummy = self.pool2(torch.relu(self.bn2(self.conv2(dummy))))
            # dummy = self.pool3(torch.relu(self.bn3(self.conv3(dummy))))
            fc_input_size = self.flatten(dummy).shape[1]

        self.fc1 = nn.Linear(fc_input_size, 64)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 64)
        self.softmax = nn.LogSoftmax(dim=1)
        self.final = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        # x = self.pool3(self.relu3(self.bn3(self.conv3(x))))

        x = self.flatten(x)
        x = self.relu1(self.fc1(x))  # Reuse relu1
        x = self.drop(x)
        x = self.relu1(self.fc2(x))
        x = self.drop(x)
        x = self.softmax(self.final(x))
        return x

holdout_data = pd.read_csv('Corpus/sturt.csv')
data = pd.read_csv('Corpus/grodner.csv') # for command line
# data = pd.read_csv('../Corpus/modified_stimulus.csv') # for pyCharm
sentences = data['Stimulus']
labels = data['Ambiguity']
# if os.path.exists("grodner_garden_path_trajectories.npy"):
#     trajectories = np.load("grodner_garden_path_trajectories.npy", allow_pickle=True)
# else:
trajectories = []
for i in range(len(sentences)):
    trajectories.append(model_runs.get_first_token_trajectory(text_utils.incrementize(sentences[i])))
    print(i + 1)

# trajectories = []
# for i in range(len(sentences)):
#     print(i + 1)
#     trajectories.append(model_runs.get_trajectory_cls(text_utils.incrementize(sentences[i])))

np.save('grodner_garden_path_trajectories.npy', np.array(trajectories, dtype=object), allow_pickle=True)
paired = [(traj, label) for traj, label in zip(trajectories, labels) if len(traj) >= 2]
trajectories_filtered, labels_filtered = list(zip(*paired))


x_train_raw, x_test_raw, y_train, y_test = train_test_split(
    list(trajectories_filtered), list(labels_filtered), test_size=0.2, random_state=42
)

x_train = vectors_between(x_train_raw)  # list of [step0, step1], each shape (seq_len, 768)
x_test = vectors_between(x_test_raw)




# holdout dataset


sentences_holdout = holdout_data['Stimulus']
labels_holdout = holdout_data['Ambiguity']
# if os.path.exists("sturt_garden_path_trajectories.npy"):
#     trajectories_holdout = np.load("sturt_garden_path_trajectories.npy", allow_pickle=True)
# else:
trajectories_holdout = []
for i in range(len(sentences)):
    trajectories_holdout.append(model_runs.get_first_token_trajectory(text_utils.incrementize(sentences[i])))
    print(i + 1)

# trajectories = []
# for i in range(len(sentences)):
#     print(i + 1)
#     trajectories.append(model_runs.get_trajectory_cls(text_utils.incrementize(sentences[i])))

np.save('sturt_garden_path_trajectories.npy', np.array(trajectories, dtype=object), allow_pickle=True)
paired_holdout = [(traj, label) for traj, label in zip(trajectories, labels) if len(traj) >= 2]
trajectories_filtered_holdout, labels_filtered_holdout = list(zip(*paired_holdout))

x_hold_out = vectors_between(trajectories_filtered_holdout)  # list of [step0, step1], each shape (seq_len, 768)
y_hold_out = labels_filtered_holdout

le = LabelEncoder()

y_hold_out = le.fit_transform(y_hold_out)


#==============================================================================


# x_train = calc_distances(x_train_raw)
# x_test = calc_distances(x_test_raw)


def get_max_len(x):
    return max((len(traj) for traj in x), default=0)


max_len = max(get_max_len(x_train), get_max_len(x_test), get_max_len(x_hold_out))

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
cnn = SimpleCNN(seq_len=max_len, num_classes=num_classes)

def to_tensor(x, desired_length):
    padded = []
    for traj in x:
        traj = np.array(traj)
        pad_len = desired_length - len(traj)
        if pad_len > 0:
            traj = np.vstack([traj, np.zeros((pad_len, traj.shape[1]))])
        else:
            traj = traj[:desired_length]
        padded.append(traj)
    return torch.tensor(np.array(padded), dtype=torch.float32)


x_train_tensor = to_tensor(x_train, max_len).permute(0, 2, 1)
x_test_tensor = to_tensor(x_test, max_len).permute(0, 2, 1)

y_train_tensor = torch.tensor(y_train_enc, dtype=torch.long)
y_test_tensor = torch.tensor(y_test_enc, dtype=torch.long)

x_hold_out_tensor = to_tensor(x_hold_out, max_len).permute(0, 2, 1)
y_hold_out_tensor = torch.tensor(y_hold_out, dtype=torch.long)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
holdout_dataset = TensorDataset(x_hold_out_tensor, y_hold_out_tensor)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
holdout_loader = DataLoader(holdout_dataset, batch_size=8, shuffle=True)

val_accuracy_list = []
train_accuracy_list = []
num_runs = 100
epochs = 25
holdout_accuracy_list = []
for i in range(num_runs):
    cnn = SimpleCNN(seq_len=max_len, num_classes=num_classes)
    acc, loss, t, accuracies, train_accuracies = train_model(cnn, train_loader, val_loader, epochs=epochs, name="CNN", lr=1e-3)
    print(acc)
    print(loss)
    print(t)
    val_accuracy_list.append(accuracies)
    train_accuracy_list.append(train_accuracies)
    holdout_accuracy = eval_model(cnn, holdout_loader, name = 'CNN')
    holdout_accuracy_list.append(holdout_accuracy)
    print(f'model accuracy on hold out dataset: {holdout_accuracy}')


val_mean_list = []
val_error_list = []

for i in range(25):
    vals = [val_accuracy_list[j][i] for j in range(num_runs)]

    val_mean_list.append(np.mean(vals))
    val_error_list.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))

train_mean_list = []
train_error_list = []

for i in range(25):
    vals = [train_accuracy_list[j][i] for j in range(num_runs)]

    train_mean_list.append(np.mean(vals))
    train_error_list.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))




# for i in range(num_runs):
#     plt.plot(train_accuracy_list[i], color='orange')
#     plt.plot(val_accuracy_list[i], color = 'blue')

plt.plot(val_mean_list, color="blue", label="Average Validation Accuracy")
plt.plot(train_mean_list, color="red", label="Average Training Accuracy")
epochs_list = np.arange(epochs)
plt.fill_between(epochs_list,
                 np.array(train_mean_list) - np.array(train_error_list),
                 np.array(train_mean_list) + np.array(train_error_list), alpha=0.2, color='red')
plt.fill_between(epochs_list,
                 np.array(val_mean_list) - np.array(val_error_list),
                 np.array(val_mean_list) + np.array(val_error_list), alpha=0.2, color = 'blue')
plt.title(f"Train and Validation Accuracy Over Epochs (Average Over {num_runs} Runs)")
plt.xlabel("Epochs")
plt.ylabel("% Accuracy")
plt.legend()
plt.ylim(40, 102)
plt.savefig("garden_path/saved_figs/garden_path.png")
plt.show()

holdout_accuracy_mean = 0
for num in holdout_accuracy_list:
    holdout_accuracy_mean += num

holdout_accuracy_mean /= len(holdout_accuracy_list)

print(f"FINAL MEAN ACCURACY ON HOLDOUT DATASET ACROSS {num_runs} RANDOM MODEL INSTANTIATIONS: {holdout_accuracy_mean}")



# SOME NOTES:
# best results from conv filters in the following order: 5 -> 5 -> 1 or 5 -> 3 -> 1