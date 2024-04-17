import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import time

# Function to check if multiple GPUs are available
def get_device():
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print(f'Using {torch.cuda.device_count()} GPUs')
            return torch.device("cuda:0")  # Change to "cuda" for automatic device selection
        else:
            return torch.device("cuda")
    else:
        return torch.device("cpu")

# Check device
device = get_device()
print("Using device:", device)

# Generate or load your dataset (assuming you have it as numpy arrays x and y)

# Apply Min-Max scaling to input data
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

# Split the scaled dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# Convert numpy arrays to PyTorch tensors
x_train_tensor = torch.from_numpy(x_train).float()
y_train_tensor = torch.from_numpy(y_train).float()
x_test_tensor = torch.from_numpy(x_test).float()
y_test_tensor = torch.from_numpy(y_test).float()

# Create TensorDatasets
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

# Create DataLoaders with a suitable batch size
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)

# Define the linear regression model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# Instantiate the model
model = LinearRegression()

# Use DataParallel for multiple GPUs
if torch.cuda.device_count() > 1:
    print(f'Using {torch.cuda.device_count()} GPUs for data parallelism')
    model = nn.DataParallel(model)

# Move the model to GPU(s)
model.to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent optimizer

# Training loop
num_epochs = 100

start_time = time.time()  # Start time
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader.dataset):.4f}')

end_time = time.time()  # End time
runtime = end_time - start_time
print(f'Training Runtime: {runtime:.2f} seconds')

# Calculate validation loss
with torch.no_grad():
    running_val_loss = 0.0
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        val_loss = criterion(outputs, labels)
        running_val_loss += val_loss.item() * inputs.size(0)

    print(f'Validation Loss: {running_val_loss / len(test_loader.dataset):.4f}')

# Test the model
x_new = torch.tensor([[6.0]]).float().to(device)
predicted = model(x_new)
print(f'Prediction after training: {predicted.item():.4f}')
