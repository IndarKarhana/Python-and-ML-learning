import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

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

# Function to preprocess data and split into train and test sets
def preprocess_data_and_split(X, y, test_size=0.2, random_state=42):
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Normalize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    # Get unique SP IDs and create SP ID tensors
    unique_sp_ids_train = X_train['sp_id'].unique()
    sp_id_mapping_train = {id_: idx for idx, id_ in enumerate(unique_sp_ids_train)}
    sp_id_train_tensor = torch.tensor([sp_id_mapping_train[id_] for id_ in X_train['sp_id']], dtype=torch.long)

    unique_sp_ids_test = X_test['sp_id'].unique()
    sp_id_mapping_test = {id_: idx for idx, id_ in enumerate(unique_sp_ids_test)}
    sp_id_test_tensor = torch.tensor([sp_id_mapping_test[id_] for id_ in X_test['sp_id']], dtype=torch.long)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, sp_id_train_tensor, sp_id_test_tensor

# Function to evaluate the model on the test set and calculate R-squared score
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        targets_all = []
        outputs_all = []
        for inputs, targets, ids in test_loader:
            inputs, targets, ids = inputs.to(device), targets.to(device), ids.to(device)
            outputs = model(inputs, ids)
            total_loss += criterion(outputs, targets.view(-1, 1)).item()
            targets_all.extend(targets.cpu().numpy())
            outputs_all.extend(outputs.cpu().numpy())

        mse = total_loss / len(test_loader)
        r2 = r2_score(targets_all, outputs_all)
    return mse, r2

# Main function
def main(X, y, hidden_size, embedding_size, num_epochs, batch_size, test_size=0.2, random_state=42):
    # Preprocess data and split into train and test sets
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, sp_id_train_tensor, sp_id_test_tensor = \
        preprocess_data_and_split(X, y, test_size=test_size, random_state=random_state)

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move data to device
    X_train_tensor = X_train_tensor.to(device)
    X_test_tensor = X_test_tensor.to(device)
    y_train_tensor = y_train_tensor.to(device)
    y_test_tensor = y_test_tensor.to(device)
    sp_id_train_tensor = sp_id_train_tensor.to(device)
    sp_id_test_tensor = sp_id_test_tensor.to(device)

    # Determine input size dynamically
    input_size = X_train_tensor.shape[1]
    num_sp_ids_train = len(sp_id_train_tensor.unique())
    num_sp_ids_test = len(sp_id_test_tensor.unique())

    # Define model
    model = NeuralNetWithUserEmbeddings(input_size, hidden_size, 1, max(num_sp_ids_train, num_sp_ids_test), embedding_size).to(device)

    # If multiple GPUs are available, wrap the model with DataParallel
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create DataLoader for training
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor, sp_id_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Train the model
    for epoch in range(num_epochs):
        model.train()
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

    # Create DataLoader for testing
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor, sp_id_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Evaluate the model
    mse, r2 = evaluate_model(model, test_loader, criterion, device)
    print(f'Test MSE: {mse:.4f}, Test R-squared: {r2:.4f}')

# Sample data
data = {
    'feature1': np.random.rand(1000),
    'feature2': np.random.rand(1000),
    'sp_id': np.random.randint(1, 101, 1000),  # Assuming 100 unique SP IDs
}
X = pd.DataFrame(data)
y = pd.Series(np.random.rand(1000))

# Hyperparameters
hidden_size = 64
embedding_size = 10
num_epochs = 10
batch_size = 32

# Call the main function
main(X, y, hidden_size, embedding_size, num_epochs, batch_size)
