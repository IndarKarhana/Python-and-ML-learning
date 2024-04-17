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

# Function to preprocess data
def preprocess_data(X, y):
    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert data to PyTorch tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)
    
    # Get unique IDs and create ID tensor
    unique_ids = X['ID'].unique()
    id_mapping = {id_: idx for idx, id_ in enumerate(unique_ids)}
    id_tensor = torch.tensor([id_mapping[id_] for id_ in X['ID']], dtype=torch.long)

    return X_tensor, y_tensor, id_tensor

# Define the architecture of the neural network with user-specific embeddings
class NeuralNetWithUserEmbeddings(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_ids, embedding_size):
        super(NeuralNetWithUserEmbeddings, self).__init__()
        self.user_embedding = nn.Embedding(num_ids, embedding_size)
        self.fc1 = nn.Linear(input_size + embedding_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, id_tensor):
        user_emb = self.user_embedding(id_tensor)
        x = torch.cat([x, user_emb], dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for inputs, targets, ids in train_loader:
            inputs, targets, ids = inputs.to(device), targets.to(device), ids.to(device)

            # Forward pass
            outputs = model(inputs, ids)

            # Compute loss
            loss = criterion(outputs, targets.view(-1, 1))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Function to evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for inputs, targets, ids in test_loader:
            inputs, targets, ids = inputs.to(device), targets.to(device), ids.to(device)
            outputs = model(inputs, ids)
            total_loss += criterion(outputs, targets.view(-1, 1)).item()

        mse = total_loss / len(test_loader)
    return mse

# Main function
def main(X, y, hidden_size, embedding_size, num_epochs, batch_size):
    # Preprocess data
    X_tensor, y_tensor, id_tensor = preprocess_data(X, y)

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move data to device
    X_tensor = X_tensor.to(device)
    y_tensor = y_tensor.to(device)
    id_tensor = id_tensor.to(device)

    # Determine input size dynamically
    input_size = X_tensor.shape[1]
    num_ids = len(id_tensor.unique())

    # Define model
    model = NeuralNetWithUserEmbeddings(input_size, hidden_size, 1, num_ids, embedding_size).to(device)

    # If multiple GPUs are available, wrap the model with DataParallel
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create DataLoader
    dataset = TensorDataset(X_tensor, y_tensor, id_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs)

    # Evaluate the model
    mse = evaluate_model(model, train_loader)
    print(f'MSE: {mse:.4f}')

# Example usage
# Load your data
# X = pd.read_csv("your_data.csv")
# y = X['target_column']
# X.drop(columns=['target_column'], inplace=True)

# Example data generation (remove this if you have your own data)
num_rows = 500000
num_features = 10
num_unique_ids = 100
X = pd.DataFrame({
    'ID': np.random.randint(0, num_unique_ids, num_rows),
    **{f'feature_{i}': np.random.randn(num_rows) for i in range(num_features)}
})
y = pd.Series(np.random.randn(num_rows))

# Set hyperparameters
hidden_size = 64
embedding_size = 10
num_epochs = 10
batch_size = 32

# Call the main function
main(X, y, hidden_size, embedding_size, num_epochs, batch_size)
