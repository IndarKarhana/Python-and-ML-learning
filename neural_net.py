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
    train_model(model, train_loader, criterion, optimizer, num_epochs)

    # Create DataLoader for testing
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor, sp_id_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Evaluate the model
    mse, r2 = evaluate_model(model, test_loader)
    print(f'Test MSE: {mse:.4f}, Test R-squared: {r2:.4f}')

# Example usage
# Load your data
# X = pd.read_csv("your_data.csv")
# y = X['target_column']
# X.drop(columns=['target_column'], inplace=True)

# Example data generation (remove this if you have your own data)
num_rows = 500000
num_features = 10
num_unique_sp_ids = 100
X = pd.DataFrame({
    'sp_id': np.random.randint(0, num_unique_sp_ids, num_rows),
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
