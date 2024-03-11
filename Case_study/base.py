# Import necessary libraries

import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split

# Define Constants
DATA_PATH = "./deep-learning-for-msc-202324/"
labels_train_path = DATA_PATH + "labels_train.csv"
sample_path = DATA_PATH + "sample.csv"
seqs_test_path = DATA_PATH + "seqs_test.csv"
seqs_train_path = DATA_PATH + "seqs_train.csv"
train_path = DATA_PATH + "train"
test_path = DATA_PATH + "test"

# Define a mapping from amino acid characters to integers
amino_acid_mapping = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
    'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
    'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
    'X': 20,  # Typically used for unknown amino acids
    'B': 21,  # Asparagine or Aspartic acid
    'Z': 22,  # Glutamine or Glutamic acid
    'J': 23,  # Leucine or Isoleucine
    '-': 24,  # Gap or padding
}

sec_struct_mapping = {'H': 0, 'E': 1, 'C': 2}  # Add more mappings if there are more labels


class ProteinDataset(Dataset):
    def __init__(self, csv_file, train_dir, label_file=None, normalize_method='min-max'):

        # Load the sequences
        self.seqs = pd.read_csv(csv_file)

        # Load the protein data from the directory
        self.protein_data = {}
        for filename in os.listdir(train_dir):
            if filename.endswith(".csv"):  # Check if the file is a CSV
                protein_id = re.split(r'_train|_test', filename)[0]
                self.protein_data[protein_id] = pd.read_csv(os.path.join(train_dir, filename))

        # Load the labels, if provided
        if label_file:
            self.labels = pd.read_csv(label_file)
        else:
            self.labels = None

        # Amino acid mapping
        self.amino_acid_mapping = amino_acid_mapping
        self.normalize_method = normalize_method

    def encode_sequence(self, sequence):
        # Convert each amino acid in the sequence to a one-hot encoded vector
        encoded_sequence = np.zeros((len(sequence), len(self.amino_acid_mapping)), dtype=int)
        for i, amino_acid in enumerate(sequence):
            # Default to 'X' for unknown amino acids
            index = self.amino_acid_mapping.get(amino_acid, self.amino_acid_mapping['X'])
            encoded_sequence[i, index] = 1
        return encoded_sequence

    def normalize_pssm(self, pssm):
        # Assuming the first two columns are non-numeric; adjust as necessary based on your actual data format
        numeric_columns = pssm[:, 2:]  # Adjust this if your numeric data starts from a different column

        # Convert to floats
        try:
            pssm_numeric = numeric_columns.astype(np.float32)
        except ValueError as e:
            # Handle or log the error if needed
            raise ValueError(f"Error converting PSSM to float: {e}")

        if self.normalize_method == 'min-max':
            # Min-Max normalization
            pssm_min = pssm_numeric.min(axis=0)
            pssm_max = pssm_numeric.max(axis=0)
            # Ensure no division by zero
            pssm_range = np.where(pssm_max - pssm_min == 0, 1, pssm_max - pssm_min)
            normalized_pssm = (pssm_numeric - pssm_min) / pssm_range
        elif self.normalize_method == 'z-score':
            # Z-Score normalization
            pssm_mean = pssm_numeric.mean(axis=0)
            pssm_std = pssm_numeric.std(axis=0)
            # Avoid division by zero
            pssm_std = np.where(pssm_std == 0, 1, pssm_std)
            normalized_pssm = (pssm_numeric - pssm_mean) / pssm_std
        else:
            # If no normalization method provided, return the original PSSM
            normalized_pssm = pssm_numeric

        return normalized_pssm

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        protein_id = self.seqs.iloc[idx]['PDB_ID']
        sequence = self.seqs.iloc[idx]['SEQUENCE']
        encoded_sequence = self.encode_sequence(sequence)  # Encode the sequence
        pssm = self.protein_data[protein_id].values  # Assuming you will process PSSM separately
        normalized_pssm = self.normalize_pssm(pssm)  # Ensure this is uncommented to use normalized PSSM

        if self.labels is not None:
            label_seq = self.labels.iloc[idx]['SEC_STRUCT']
            label_numeric = [sec_struct_mapping[char] for char in label_seq]
            label_tensor = torch.tensor(label_numeric, dtype=torch.long)
            return (
                protein_id,
                torch.tensor(encoded_sequence, dtype=torch.float32),
                torch.tensor(normalized_pssm, dtype=torch.float32),
                label_tensor
            )

        return (
            protein_id,
            torch.tensor(encoded_sequence, dtype=torch.float32),
            torch.tensor(normalized_pssm, dtype=torch.float32)
        )


def collate_fn_without_labels(batch):
    id, sequences, pssms = zip(*batch)  # Unzip the batch

    # Pad sequences and PSSMs
    sequences_padded = pad_sequence([seq.clone().detach() for seq in sequences], batch_first=True)
    # sequences_padded =  torch.tensor([seq.clone().detach() for seq in sequences])

    pssms_padded = torch.tensor(pssms)
    # pssms_padded = pad_sequence([pssm.clone().detach() for pssm in pssms], batch_first=True)

    return id, sequences_padded, pssms_padded


def collate_fn(batch):
    _, sequences, pssms, labels_list = zip(*batch)  # Unzip the batch

    # Pad sequences and PSSMs
    sequences_padded = pad_sequence([seq.clone().detach() for seq in sequences], batch_first=True)

    pssms_padded = pad_sequence([pssm.clone().detach() for pssm in pssms], batch_first=True)

    # Handling labels correctly
    if labels_list[0] is not None:  # Check if labels exist
        labels_padded = pad_sequence([label.clone().detach() for label in labels_list], batch_first=True)

    else:
        labels_padded = None

    # Create a mask based on the original sequence lengths
    mask = [torch.ones(len(label), dtype=torch.uint8) for label in labels_list]
    mask_padded = pad_sequence(mask, batch_first=True, padding_value=0)  # Assuming padding_value for labels is 0
    return sequences_padded, pssms_padded, labels_padded, mask_padded


class FullyConvolutionalProteinModel(nn.Module):
    def __init__(self, num_classes=3, input_channels=20):  # 20 for amino acid one-hot, adjust if using PSSM
        super(FullyConvolutionalProteinModel, self).__init__()

        # Define convolutional layers
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)

        # Final layer that maps to the number of classes
        self.final_conv = nn.Conv1d(in_channels=256, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        # Apply convolutional layers with activation functions
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Apply final convolutional layer - no activation, as CrossEntropyLoss includes it
        x = self.final_conv(x)

        # No softmax here, as nn.CrossEntropyLoss applies it internally.
        # Transpose the output to match [batch_size, sequence_length, num_classes]
        # This makes it easier to calculate loss later
        x = x.transpose(1, 2)

        return x


def train_model(model, criterion, optimizer, train_dataloader, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for sequences, pssms, labels, _ in train_dataloader:
            inputs = pssms.permute(0, 2, 1)  # Adjust for PSSM data

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs.transpose(1, 2), labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            # Calculate training accuracy
            _, predicted = torch.max(outputs, 2)  # Get the index of the max log-probability
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.numel()

        epoch_loss = running_loss / len(train_dataloader.dataset)
        epoch_acc = correct_preds / total_preds
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')


def validate_model(model, criterion, val_dataloader):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for sequences, pssms, labels, _ in val_dataloader:
            inputs = pssms.permute(0, 2, 1)  # Adjust for PSSM data

            outputs = model(inputs)
            loss = criterion(outputs.transpose(1, 2), labels)

            running_loss += loss.item() * inputs.size(0)

            # Calculate accuracy
            _, predicted = torch.max(outputs, 2)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.numel()

    val_loss = running_loss / len(val_dataloader.dataset)
    val_acc = correct_preds / total_preds
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')


def test_model(model, criterion, test_dataloader):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for sequences, pssms, labels, _ in test_dataloader:
            inputs = pssms.permute(0, 2, 1)  # Adjust for PSSM data

            outputs = model(inputs)
            loss = criterion(outputs.transpose(1, 2), labels)

            running_loss += loss.item() * inputs.size(0)

            # Calculate accuracy
            _, predicted = torch.max(outputs, 2)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.numel()

    test_loss = running_loss / len(test_dataloader.dataset)
    test_acc = correct_preds / total_preds
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')


def test_model_direct(model, test_dataset, output_file='./submission.csv'):
    model.eval()  # Set the model to evaluation mode
    predictions = []

    with torch.no_grad():
        for i in range(len(test_dataset)):  # Iterate directly over the dataset
            pdb_id, _, pssm = test_dataset[i]  # Assuming the dataset returns PDB_ID, sequence, and PSSM

            # Prepare the input tensor; add an extra batch dimension using unsqueeze
            input_pssm = pssm.unsqueeze(0).permute(0, 2, 1)  # Adjust dimensions to [1, features, seq_len]

            # Make a prediction
            outputs = model(input_pssm)
            _, predicted = torch.max(outputs, 2)  # Get the index of max log-probability

            # Process the predictions
            seq_len = pssm.shape[0]  # Assuming pssm is [features, seq_len]
            for j in range(seq_len):
                residue_id = f"{pdb_id}_{j + 1}"  # Construct the ID
                structure_label = ['H', 'E', 'C'][predicted[0, j].item()]  # Map numeric predictions to labels
                predictions.append([residue_id, structure_label])

    # Write predictions to CSV
    pd.DataFrame(predictions, columns=['ID', 'STRUCTURE']).to_csv(output_file, index=False)
    print(f'Submission file saved to {output_file}')


dataset = ProteinDataset(csv_file=seqs_train_path, train_dir=train_path, label_file=labels_train_path)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

test_dataset = ProteinDataset(csv_file=seqs_test_path, train_dir=test_path)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn_without_labels)

# Splitting train_dataset into train and validation sets (adjust sizes as needed)
# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size
# train_subset, val_subset = random_split(dataset, [train_size, val_size])

# val_loader = DataLoader(val_subset, batch_size=64, shuffle=False, collate_fn=collate_fn)

# model = FullyConvolutionalProteinModel(hidden_layers=[64, 128, 256, 512, 1024])
model = FullyConvolutionalProteinModel()
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.000957)  # Learning rate may need tuning
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, weight_decay=0.0)
num_epochs = 10

train_model(model, criterion, optimizer, dataloader)
# validate_model(model, criterion, val_loader)
# test_model_to_csv(model, test_loader)
# test_model_to_csv_test(None, test_loader)

test_model_direct(model, test_dataset)
