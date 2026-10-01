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


class MLP(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(768, 64),
            nn.ReLU(),
            nn.Dropout(0.1),

            # nn.Linear(64, 32),
            # nn.ReLU(),
            # nn.Dropout(0.3),

            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)


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
    trajectories.append(model_runs.get_last_token_trajectory(text_utils.incrementize(sentences[i])))
    print(i + 1)

print("TRAJECTORIES: ", trajectories)
# trajectories = []
# for i in range(len(sentences)):
#     print(i + 1)
#     trajectories.append(model_runs.get_trajectory_cls(text_utils.incrementize(sentences[i])))

np.save('grodner_garden_path_trajectories.npy', np.array(trajectories, dtype=object), allow_pickle=True)
paired = [(traj, label) for traj, label in zip(trajectories, labels)]
trajectories_filtered, labels_filtered = list(zip(*paired))


x_train_raw, x_test_raw, y_train, y_test = train_test_split(
    list(trajectories_filtered), list(labels_filtered), test_size=0.2, random_state=42
)

print(x_train_raw)
x_train = x_train_raw  # list of [step0, step1], each shape (seq_len, 768)
x_test = x_test_raw




# holdout dataset


sentences_holdout = holdout_data['Stimulus']
labels_holdout = holdout_data['Ambiguity']
# if os.path.exists("sturt_garden_path_trajectories.npy"):
#     trajectories_holdout = np.load("sturt_garden_path_trajectories.npy", allow_pickle=True)
# else:
trajectories_holdout = []
for i in range(len(sentences_holdout)):
    trajectories_holdout.append(model_runs.get_last_token_trajectory(text_utils.incrementize(sentences_holdout[i])))
    print(i + 1)

# trajectories = []
# for i in range(len(sentences)):
#     print(i + 1)
#     trajectories.append(model_runs.get_trajectory_cls(text_utils.incrementize(sentences[i])))

np.save('sturt_garden_path_trajectories.npy', np.array(trajectories_holdout, dtype=object), allow_pickle=True)
paired_holdout = [(traj, label) for traj, label in zip(trajectories_holdout, labels)]
trajectories_filtered_holdout, labels_filtered_holdout = list(zip(*paired_holdout))

x_hold_out = trajectories_filtered_holdout  # list of [step0, step1], each shape (seq_len, 768)
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
cnn = MLP(num_classes=num_classes)

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


x_train_tensor = torch.tensor(np.array([traj[0] for traj in x_train]), dtype=torch.float32)
x_test_tensor = torch.tensor(np.array([traj[0] for traj in x_test]), dtype=torch.float32)

y_train_tensor = torch.tensor(y_train_enc, dtype=torch.long)
y_test_tensor = torch.tensor(y_test_enc, dtype=torch.long)

x_hold_out_tensor = torch.tensor(np.array([traj[0] for traj in x_hold_out]), dtype=torch.float32)
y_hold_out_tensor = torch.tensor(y_hold_out, dtype=torch.long)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
holdout_dataset = TensorDataset(x_hold_out_tensor, y_hold_out_tensor)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
holdout_loader = DataLoader(holdout_dataset, batch_size=8, shuffle=True)

val_accuracy_list = []
train_accuracy_list = []
num_runs = 10
epochs = 100
holdout_accuracy_list = []
for i in range(num_runs):
    cnn = MLP(num_classes=num_classes)
    acc, loss, t, accuracies, train_accuracies = train_model(cnn, train_loader, val_loader, epochs=epochs, name="CNN", lr=0.0006)
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

for i in range(100):
    vals = [val_accuracy_list[j][i] for j in range(num_runs)]

    val_mean_list.append(np.mean(vals))
    val_error_list.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))

train_mean_list = []
train_error_list = []

for i in range(100):
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
plt.ylim(0, 102)
plt.savefig("garden_path/saved_figs/garden_path_last_token.png")
plt.show()

holdout_accuracy_mean = 0
for num in holdout_accuracy_list:
    holdout_accuracy_mean += num

holdout_accuracy_mean /= len(holdout_accuracy_list)

print(f"FINAL MEAN ACCURACY ON HOLDOUT DATASET ACROSS {num_runs} RANDOM MODEL INSTANTIATIONS: {holdout_accuracy_mean}")



# SOME NOTES:
# best results from conv filters in the following order: 5 -> 5 -> 1 or 5 -> 3 -> 1