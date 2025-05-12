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
X = torch.nan_to_num(X) # Handle missing entries by zero imputation

y = torch.tensor(ds.data.targets.to_numpy(), dtype=torch.float)
y[y > 0] = 1 # Turn into binary classification task

# Split the data into training, validation, and test sets
seed = 42
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=seed)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)

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
model = MLP(
    input_size = X.shape[1],
    hidden_size = 8,  # Example hidden layer size
    output_size = y.shape[1]
)
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

# Evaluate on the test set
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
print(f'Test Loss: {test_loss.item():.4f}')
print(outputs.shape[0])
