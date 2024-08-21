import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import TensorDataset, random_split, DataLoader
import numpy as np
from tqdm.notebook import trange, tqdm
import h5py
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import matplotlib.pyplot as plt
import os

# Suppress matplotlib deprecation warnings
import warnings
from matplotlib import MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=16, stride=4, dropout_prob=0.8):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2 - 1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)

        # Shortcut with Max Pooling followed by 1x1 Conv
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.MaxPool1d(kernel_size=stride, stride=stride),  # Downsample by a factor of stride (e.g., 4)
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)  # Match the channel dimension
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)  # Apply dropout after the first convolution and activation
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(self.bn2(out))
        out = self.dropout2(out)
        return out


class AFibDetector(nn.Module):
    def __init__(self):
        super(AFibDetector, self).__init__()
        kernel_size = 16
        self.initial_conv = nn.Conv1d(in_channels=8, out_channels=64, kernel_size=kernel_size)
        self.initial_bn = nn.BatchNorm1d(64)

        # Adding 4 Residual Blocks:
        # 1. Increase the number of filters by 64, every 2nd block
        # 2. Subsample by a factor of 4 at every block
        stride = 4
        self.res_block1 = ResidualBlock(in_channels=64, out_channels=64, kernel_size=kernel_size, stride=stride)
        self.res_block2 = ResidualBlock(in_channels=64, out_channels=128, kernel_size=kernel_size, stride=stride)
        self.res_block3 = ResidualBlock(in_channels=128, out_channels=128, kernel_size=kernel_size, stride=stride)
        self.res_block4 = ResidualBlock(in_channels=128, out_channels=256, kernel_size=kernel_size, stride=stride)

        # Calculate the number of flat features after the residual blocks
        dummy_input = torch.zeros(1, 8, 4096)
        dummy_output = self.initial_conv(dummy_input)
        dummy_output = F.relu(self.initial_bn(dummy_output))
        dummy_output = self.res_block1(dummy_output)
        dummy_output = self.res_block2(dummy_output)
        dummy_output = self.res_block3(dummy_output)
        dummy_output = self.res_block4(dummy_output)
        flat_features = dummy_output.numel()

        # Fully Connected Layers
        self.fc1 = nn.Linear(flat_features, 256)
        self.output = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.initial_bn(self.initial_conv(x)))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = torch.sigmoid(self.output(x))
        return x


def train_loop(epoch, dataloader, model, optimizer, loss_function, device):
    # model to training mode (important to correctly handle dropout or batchnorm layers)
    model.train()
    # allocation
    total_loss = 0  # accumulated loss
    n_entries = 0   # accumulated number of data points
    # progress bar def
    train_pbar = tqdm(dataloader, desc="Training Epoch {epoch:2d}".format(epoch=epoch), leave=True)
    # training loop
    for traces, diagnoses in train_pbar:
        # data to device (CPU or GPU if available)
        traces, diagnoses = traces.to(device), diagnoses.to(device)

        # 1. Zero the gradients
        optimizer.zero_grad()
        # 2. Forward pass: compute the model's predictions for the input batch
        predictions = model(traces)
        # 3. Compute loss: compare the model output with the true labels
        loss = loss_function(predictions, diagnoses)
        # 4. Backward pass: compute the gradient of the loss with respect to model parameters
        loss.backward()
        # 5. Optimization step: update the model's parameters
        optimizer.step()
        
        # Update accumulated values
        total_loss += loss.detach().cpu().numpy()
        n_entries += len(traces)

        # Update progress bar
        train_pbar.set_postfix({'loss': total_loss / n_entries})
    train_pbar.close()
    return total_loss / n_entries


def eval_loop(epoch, dataloader, model, loss_function, device):
    # model to evaluation mode (important to correctly handle dropout or batchnorm layers)
    model.eval()
    # allocation
    total_loss = 0  # accumulated loss
    n_entries = 0   # accumulated number of data points
    valid_probs = []  # accumulated predicted probabilities
    valid_true = [] # accumulated true labels
    
    # progress bar def
    eval_pbar = tqdm(dataloader, desc="Evaluation Epoch {epoch:2d}".format(epoch=epoch), leave=True)
    # evaluation loop
    for traces_cpu, diagnoses_cpu in eval_pbar:
        # data to device (CPU or GPU if available)
        traces, diagnoses = traces_cpu.to(device), diagnoses_cpu.to(device)

        # Disable gradient computation for efficiency and to prevent changes to the model
        with torch.no_grad():
            # Forward pass: compute the model's predictions for the input batch
            predictions = model(traces)

            # Compute loss: compare the model output with the true labels
            loss = loss_function(predictions, diagnoses)

            # Store probabilities and true labels for later analysis
            valid_probs.append(predictions.cpu().numpy())
            valid_true.append(diagnoses_cpu.numpy())  # Use CPU data directly to avoid unnecessary GPU-to-CPU transfer

        # Update accumulated values
        total_loss += loss.detach().cpu().numpy()
        n_entries += len(traces)

        # Update progress bar
        eval_pbar.set_postfix({'loss': total_loss / n_entries})
    eval_pbar.close()
    return total_loss / n_entries, np.vstack(valid_probs), np.vstack(valid_true)


def train(config_name):
    tqdm.write(f"Training {config_name}")
    best_loss = np.Inf
    train_loss_all, valid_loss_all = [], []
    auroc_all, accuracy_all, f1_all = [], [], []

    # loop over epochs
    for epoch in trange(1, num_epochs + 1):
        # training loop
        train_loss = train_loop(epoch, train_dataloader, model, optimizer, loss_function, device)
        # validation loop
        valid_loss, y_pred, y_true = eval_loop(epoch, valid_dataloader, model, loss_function, device)

        # collect losses
        train_loss_all.append(train_loss)
        valid_loss_all.append(valid_loss)

        # Use the raw output as it already contains probabilities
        y_pred = torch.tensor(y_pred).numpy()
        auroc = roc_auc_score(y_true, y_pred)
        auroc_all.append(auroc)

        # Convert probabilities to binary predictions
        y_pred_labels = (y_pred >= 0.5).astype(int)
        accuracy = accuracy_score(y_true, y_pred_labels)
        accuracy_all.append(accuracy)
        f1 = f1_score(y_true, y_pred_labels)
        f1_all.append(f1)

        # save best model: here we save the model only for the lowest validation loss
        if valid_loss < best_loss:
            # Save model parameters
            torch.save({'model': model.state_dict()}, 'model.pth')
            # Update best validation loss
            best_loss = valid_loss
            # statement
            model_save_state = "Best model"
        else:
            model_save_state = ""

        # Print message
        tqdm.write('Epoch {epoch:2d}: \t'
                   'Train Loss {train_loss:.6f} \t'
                   'Valid Loss {valid_loss:.6f} \t'
                   'F1 score {f1:.3f} \t'
                   'AUROC {auroc:.4f} \t'
                   '{model_save_state}'
                   .format(epoch=epoch,
                           train_loss=train_loss,
                           valid_loss=valid_loss,
                           f1=f1,
                           auroc=auroc,
                           model_save_state=model_save_state))

        # Update learning rate with lr-scheduler
        sched.step(valid_loss)

    # rename best model
    os.rename("model.pth", f"{config_name}.pth")

    # Calculate the best points
    best_valid_loss_idx = valid_loss_all.index(min(valid_loss_all))
    best_auroc_idx = auroc_all.index(max(auroc_all))
    best_accuracy_idx = accuracy_all.index(max(accuracy_all))
    best_f1_idx = f1_all.index(max(f1_all))

    # Clear the current figure
    plt.clf()

    # Plot Losses
    plt.subplot(2, 2, 1)
    plt.plot(train_loss_all, label='Train Loss')
    plt.plot(valid_loss_all, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.annotate(f'{valid_loss_all[best_valid_loss_idx]:.6f}',
                 xy=(best_valid_loss_idx, valid_loss_all[best_valid_loss_idx]),
                 xytext=(best_valid_loss_idx, valid_loss_all[best_valid_loss_idx]),
                 arrowprops=dict(facecolor='black', arrowstyle='->'))

    # Plot AUROC
    plt.subplot(2, 2, 2)
    plt.plot(auroc_all, label='Validation AUROC')
    plt.title('AUROC over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('AUROC')
    plt.legend()
    plt.annotate(f'{auroc_all[best_auroc_idx]:.4f}',
                 xy=(best_auroc_idx, auroc_all[best_auroc_idx]),
                 xytext=(best_auroc_idx, auroc_all[best_auroc_idx]),
                 arrowprops=dict(facecolor='black', arrowstyle='->'))

    # Plot Accuracy
    plt.subplot(2, 2, 3)
    plt.plot(accuracy_all, label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.annotate(f'{accuracy_all[best_accuracy_idx]:.4f}',
                 xy=(best_accuracy_idx, accuracy_all[best_accuracy_idx]),
                 xytext=(best_accuracy_idx, accuracy_all[best_accuracy_idx]),
                 arrowprops=dict(facecolor='black', arrowstyle='->'))

    # Plot F1 Score
    plt.subplot(2, 2, 4)
    plt.plot(f1_all, label='Validation F1 Score')
    plt.title('F1 Score over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.annotate(f'{f1_all[best_f1_idx]:.4f}',
                 xy=(best_f1_idx, f1_all[best_f1_idx]),
                 xytext=(best_f1_idx, f1_all[best_f1_idx]),
                 arrowprops=dict(facecolor='black', arrowstyle='->'))

    plt.tight_layout()
    plt.savefig(f"{config_name}.pdf")

    # Optionally, you can close the plot after saving it
    plt.close()


if __name__ == '__main__':

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tqdm.write("Use device: {device:}\n".format(device=device))

    # set seed
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    tqdm.write("Define model...")
    model = AFibDetector()
    model.to(device=device)
    tqdm.write("Done!\n")

    # load traces
    path_to_h5_train, path_to_csv_train, path_to_records = ('codesubset/train.h5', 'codesubset/train.csv',
                                                            'codesubset/train/RECORDS.txt')
    traces = torch.tensor(h5py.File(path_to_h5_train, 'r')['tracings'][()],
                          dtype=torch.float32).transpose(1, 2)

    # load labels
    ids_traces = [int(x.split('TNMG')[1]) for x in
                  list(pd.read_csv(path_to_records, header=None)[0])]  # Get order of ids in traces
    df = pd.read_csv(path_to_csv_train)
    df.set_index('id_exam', inplace=True)
    df = df.reindex(ids_traces)  # make sure the order is the same
    labels = torch.tensor(np.array(df['AF']), dtype=torch.float32).reshape(-1, 1)

    # load dataset
    dataset = TensorDataset(traces, labels)
    len_dataset = len(dataset)
    n_classes = len(torch.unique(labels))

    # split data
    split_ratio = 0.95
    split_train = int(len_dataset * split_ratio)
    split_valid = len_dataset - split_train
    dataset_train, dataset_valid = random_split(dataset, [split_train, split_valid])

    # define loss function
    # Since our dataset is imbalanced 7-3 in favor of negatives, let's use a pos weight set to their ratio 7/3
    pos_weight = torch.tensor([7 / 3]).to(device)
    loss_function = nn.BCELoss(pos_weight)

    # choose hyperparameters
    weight_decay = 1e-1
    num_epochs = 80

    for batch_size in [8, 16, 32, 64, 128]:
        # build data loaders
        train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)

        for learning_rate in [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]:

            # optimizer and scheduler
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            sched = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

            lr_to_str = f"{learning_rate:.5f}".replace(".", "_")
            train(f"lr_{lr_to_str}_bs_{batch_size}")
