import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from MLP import MLP
from DataReader import *
import random

data_dir = "./data/"
train_filename = "training.npz"
test_filename = "test.npz"

# set seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Hyperparameters
EPOCHS = 50
LEARNING_RATE = 0.001
HIDDEN_SIZE = 3
HIDDEN_SIZES = [3, 10, 20, 50, 100]
BATCH_SIZE = 32

def evaluation(model, best_model_state, best_val_acc):
    ### YOUR CODE HERE
    # step 1: Load test data
    raw_test_data, test_labels = load_data(os.path.join(data_dir, test_filename))

    # step 2: Preprocess raw data to extract features
    test_X = prepare_X(raw_test_data)
    test_y, _ = prepare_y(test_labels)

    # step 3: Convert numpy arrays to PyTorch tensors
    test_X = torch.tensor(test_X, dtype=torch.float32)
    test_y = np.array(test_y)
    test_y = torch.tensor(test_y, dtype=torch.long)

    # step 4: Load the best validation model
    model.load_state_dict(best_model_state)
    model.eval()

    # step 5: Evaluate classification accuracy on the test set
    with torch.no_grad():
        outputs = model(test_X)
        predictions = torch.argmax(outputs, dim=1)
        test_acc = (predictions == test_y).float().mean().item()

    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    return test_acc


    ### END YOUR CODE

##### Load Data
raw_data, labels = load_data(os.path.join(data_dir, train_filename))
raw_train, raw_valid, label_train, label_valid = train_valid_split(raw_data, labels, 2300)

##### Preprocess raw data to extract features
train_X_all = prepare_X(raw_train)
valid_X_all = prepare_X(raw_valid)

##### Preprocess labels for all three classes (0,1,2)
train_y_all, _ = prepare_y(label_train)
valid_y_all, _ = prepare_y(label_valid)


##### Convert numpy array to torch tensors
train_X_all = torch.tensor(train_X_all, dtype=torch.float32)
valid_X_all = torch.tensor(valid_X_all, dtype=torch.float32)

train_y_all = torch.tensor(train_y_all, dtype=torch.long)
valid_y_all = torch.tensor(valid_y_all, dtype=torch.long)


# Initialize model, loss, and optimizer
model = MLP(input_size=train_X_all.shape[1], hidden_size=HIDDEN_SIZE, output_size=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Track the best validation accuracy and corresponding model
best_val_acc = 0.0
best_model_state = None

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    num_batches = len(train_X_all) // BATCH_SIZE

    # Mini-batch training (manual batching)
    for i in range(num_batches):
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, len(train_X_all))

        X_batch = train_X_all[start_idx:end_idx]
        y_batch = train_y_all[start_idx:end_idx]

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        outputs = model(valid_X_all)
        predictions = torch.argmax(outputs, dim=1)
        val_acc  = (predictions == valid_y_all).float().mean().item()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss / num_batches:.4f}, Val Accuracy: {val_acc:.4f}")

    # Save the best validation model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict()


##### Test the best model on the test set
evaluation(model, best_model_state, best_val_acc)

# Save the best trained model
torch.save(best_model_state, "best_mlp_model.pth")
print("Training complete. Best model saved as best_mlp_model.pth.")


best_test_acc = 0.0
best_hidden_size = HIDDEN_SIZES[0]

for hidden_size in HIDDEN_SIZES:
    print("=============================================================================")
    print(f"Training with HIDDEN_SIZE = {hidden_size}")
    print("=============================================================================")
    model = MLP(input_size=train_X_all.shape[1], hidden_size=hidden_size, output_size=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop (same as before)
    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        num_batches = len(train_X_all) // BATCH_SIZE

        for i in range(num_batches):
            start_idx = i * BATCH_SIZE
            end_idx = min((i + 1) * BATCH_SIZE, len(train_X_all))

            X_batch = train_X_all[start_idx:end_idx]
            y_batch = train_y_all[start_idx:end_idx]

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            outputs = model(valid_X_all)
            predictions = torch.argmax(outputs, dim=1)
            val_acc = (predictions == valid_y_all).float().mean().item()

        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss / num_batches:.4f}, Val Accuracy: {val_acc:.4f}")

        # Save the best validation model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()

    # Evaluate on test set
    evaluation(model, best_model_state, best_val_acc)

    # Track the best hidden size
    if best_val_acc > best_test_acc:
        best_test_acc = best_val_acc
        best_hidden_size = hidden_size

print(f"Best Hidden Size: {best_hidden_size}, Best Test Accuracy: {best_test_acc:.4f}")