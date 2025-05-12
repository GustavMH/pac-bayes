#!/usr/bin/env python3

from uciml_fetch import fetch_ucirepo, list_available_datasets
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
# check which datasets can be imported
#list_available_datasets()

ds = fetch_ucirepo(name='Heart Disease')
X = torch.tensor(ds.data.features.to_numpy(), dtype=torch.float)
y = torch.tensor(ds.data.targets.to_numpy(), dtype=torch.float)

# access metadata
print(ds.metadata.additional_info.summary)

# access variable info in tabular format
print(ds.variables)

# Handle missing entries by zero imputation
X = torch.nan_to_num(X)

# Turn into binary classification task
y[y > 0] = 1

# Split the data into training, validation, and test sets
seed = 42
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=seed)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)

# TODO rewrite above section to pipeline

# TODO keras, Define the MLP with shortcut connections
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.shortcut = nn.Linear(input_size, output_size)

    def forward(self, x):
        out1 = torch.relu(self.fc1(x))
        out2 = torch.relu(self.fc2(out1))
        out3 = self.fc3(out2)
        shortcut = self.shortcut(x)
        return out3 + shortcut

# Define the model, loss function, and optimizer
input_size = X.shape[1]
hidden_size = 8  # Example hidden layer size
output_size = y.shape[1]
model = MLP(input_size, hidden_size, output_size)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Rprop(model.parameters())

# Training loop
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            train_acc = (torch.round(torch.nn.functional.sigmoid(outputs)) == y_train).sum() / outputs.shape[0]
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            val_acc = (torch.round(torch.nn.functional.sigmoid(val_outputs)) == y_val).sum() / val_outputs.shape[0]
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f} {train_acc.item():.4f}, Val Loss: {val_loss.item():.4f} {val_acc.item():.4f}')

# Evaluate on the test set
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
print(f'Test Loss: {test_loss.item():.4f}')
print(outputs.shape[0])
