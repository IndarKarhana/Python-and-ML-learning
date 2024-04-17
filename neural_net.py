import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import time

# Toy dataset
x = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)
y = np.array([[2.0], [4.0], [6.0], [8.0], [10.0]], dtype=np.float32)

# Scale the input features
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# Convert numpy arrays to PyTorch tensors
x_train_tensor = torch.from_numpy(x_train)
y_train_tensor = torch.from_numpy(y_train)
x_test_tensor = torch.from_numpy(x_test)
y_test_tensor = torch.from_numpy(y_test)

# Define the linear regression model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # 1 input feature, 1 output feature (1D to 1D)

    def forward(self, x):
        return self.linear(x)

# Instantiate the model
model = LinearRegression()

# Define loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent optimizer

# Training loop
num_epochs = 1000

start_time = time.time()  # Start time
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(x_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

end_time = time.time()  # End time
runtime = end_time - start_time
print(f'Training Runtime: {runtime:.2f} seconds')

# Scale the new input feature for prediction
x_new_scaled = scaler.transform(np.array([[6.0]]))
x_new = torch.from_numpy(x_new_scaled)

# Test the model
with torch.no_grad():
    predicted = model(x_new)
    print(f'Prediction after training: {predicted.item():.4f}')

# Inverse transform the predicted value to get the original scale
predicted_original_scale = scaler.inverse_transform([[predicted.item()]])
print(f'Prediction in original scale: {predicted_original_scale[0][0]:.4f}')
