import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

# Assuming you have your data stored in X and y DataFrames
# Let's create some dummy data for demonstration
num_users = 100
num_samples_per_user = 1000
num_features = 10

# Generate dummy data
X_data = np.random.randn(num_users * num_samples_per_user, num_features)
y_data = np.random.randn(num_users * num_samples_per_user)

# Convert data to DataFrames
X = pd.DataFrame(X_data, columns=[f'feature_{i}' for i in range(num_features)])
X['user_id'] = np.repeat(np.arange(num_users), num_samples_per_user)
y = pd.Series(y_data)

# Normalize the features
scaler = StandardScaler()
X.iloc[:, :-1] = scaler.fit_transform(X.iloc[:, :-1])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.iloc[:, :-1].values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.iloc[:, :-1].values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
user_ids_train = torch.tensor(X_train['user_id'].values, dtype=torch.long)
user_ids_test = torch.tensor(X_test['user_id'].values, dtype=torch.long)

# Define the architecture of the neural network with user-specific embeddings
class NeuralNetWithUserEmbeddings(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_users, embedding_size):
        super(NeuralNetWithUserEmbeddings, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.fc1 = nn.Linear(input_size + embedding_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, user_ids):
        user_emb = self.user_embedding(user_ids)
        x = torch.cat([x, user_emb], dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move data to device
X_train_tensor = X_train_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)
user_ids_train = user_ids_train.to(device)
user_ids_test = user_ids_test.to(device)

# Define model parameters
input_size = X_train_tensor.shape[1]
output_size = 1
hidden_size = 64
embedding_size = 10

# Define models
model_with_embeddings = NeuralNetWithUserEmbeddings(input_size, hidden_size, output_size, num_users, embedding_size).to(device)

# If multiple GPUs are available, wrap the model with DataParallel
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model_with_embeddings = nn.DataParallel(model_with_embeddings)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer_with_embeddings = optim.Adam(model_with_embeddings.parameters(), lr=0.001)

# Train the model
num_epochs = 10
batch_size = 32

# Training with user embeddings
for epoch in range(num_epochs):
    for i in range(0, len(X_train_tensor), batch_size):
        inputs = X_train_tensor[i:i+batch_size]
        targets = y_train_tensor[i:i+batch_size]
        users = user_ids_train[i:i+batch_size]

        # Forward pass
        outputs = model_with_embeddings(inputs, users)

        # Compute loss
        loss = criterion(outputs, targets.view(-1, 1))

        # Backward and optimize
        optimizer_with_embeddings.zero_grad()
        loss.backward()
        optimizer_with_embeddings.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss with embeddings: {loss.item():.4f}')

# Evaluate the model
model_with_embeddings.eval()
with torch.no_grad():
    outputs = model_with_embeddings(X_test_tensor, user_ids_test)
    mse = mean_squared_error(y_test_tensor.cpu().numpy(), outputs.cpu().numpy().flatten())

print(f'MSE with embeddings: {mse:.4f}')
