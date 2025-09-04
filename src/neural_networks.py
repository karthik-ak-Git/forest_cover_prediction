"""
Forest Cover Type Prediction - Neural Network Models
This module implements advanced neural network architectures for achieving 99% accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import config


class ForestCoverNet(nn.Module):
    """Advanced Neural Network for Forest Cover Type Prediction"""

    def __init__(self, input_dim, hidden_layers=[512, 256, 128, 64],
                 dropout_rate=0.3, num_classes=7):
        super(ForestCoverNet, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        # Build layers dynamically
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_layers):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            # Batch normalization
            layers.append(nn.BatchNorm1d(hidden_dim))
            # Activation
            layers.append(nn.ReLU())
            # Dropout
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights using Xavier initialization"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.network(x)


class ForestCoverNetAdvanced(nn.Module):
    """Advanced Neural Network with Residual Connections"""

    def __init__(self, input_dim, hidden_layers=[512, 256, 128, 64],
                 dropout_rate=0.3, num_classes=7):
        super(ForestCoverNetAdvanced, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_layers[0])
        self.input_bn = nn.BatchNorm1d(hidden_layers[0])

        # Hidden layers with residual connections
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_layers) - 1):
            layer = nn.Sequential(
                nn.Linear(hidden_layers[i], hidden_layers[i + 1]),
                nn.BatchNorm1d(hidden_layers[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            self.hidden_layers.append(layer)

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_layers[-1],
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )

        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_layers[-1], hidden_layers[-1] // 2),
            nn.BatchNorm1d(hidden_layers[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_layers[-1] // 2, num_classes)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        # Input projection
        x = F.relu(self.input_bn(self.input_proj(x)))

        # Hidden layers with residual connections
        residual = x
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            # Add residual connection every 2 layers
            if i > 0 and i % 2 == 0 and x.shape == residual.shape:
                x = x + residual
                residual = x

        # Self-attention (reshape for attention mechanism)
        x_reshaped = x.unsqueeze(1)  # Add sequence dimension
        attended, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
        x = attended.squeeze(1)  # Remove sequence dimension

        # Output
        return self.output_layers(x)


class NeuralNetworkTrainer:
    """Training manager for neural networks"""

    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def validate(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)

                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def train(self, train_loader, val_loader, epochs=200,
              learning_rate=0.001, patience=20, target_accuracy=99.0):
        """Complete training loop with early stopping"""

        # Setup optimizer and scheduler
        optimizer = optim.AdamW(self.model.parameters(),
                                lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0
        patience_counter = 0
        best_model_state = None

        print(f"Starting training for {epochs} epochs...")
        print(f"Target accuracy: {target_accuracy}%")

        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch(
                train_loader, optimizer, criterion)

            # Validation
            val_loss, val_acc = self.validate(val_loader, criterion)

            # Update learning rate
            scheduler.step(val_acc)

            # Save metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            # Print progress
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f'Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

            # Check for improvement
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()

                # Check if target accuracy reached
                if val_acc >= target_accuracy:
                    print(
                        f"🎯 Target accuracy {target_accuracy}% reached! Val Acc: {val_acc:.2f}%")
                    break
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                print(
                    f"Early stopping at epoch {epoch}. Best val accuracy: {best_val_acc:.2f}%")
                break

        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        print(
            f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
        return best_val_acc

    def evaluate(self, test_loader):
        """Evaluate the model on test set"""
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)

                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        accuracy = accuracy_score(all_targets, all_preds)
        report = classification_report(all_targets, all_preds)
        cm = confusion_matrix(all_targets, all_preds)

        return accuracy, report, cm, all_preds, all_targets

    def predict(self, data_loader):
        """Make predictions"""
        self.model.eval()
        predictions = []
        probabilities = []

        with torch.no_grad():
            for data in data_loader:
                if isinstance(data, tuple):  # If data has targets
                    data = data[0]
                data = data.to(self.device)
                output = self.model(data)

                # Get probabilities
                probs = F.softmax(output, dim=1)
                probabilities.extend(probs.cpu().numpy())

                # Get predictions
                pred = output.argmax(dim=1)
                predictions.extend(pred.cpu().numpy())

        return np.array(predictions), np.array(probabilities)


def create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=512):
    """Create PyTorch data loaders"""

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(
        y_train - 1)  # Convert to 0-based indexing
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val - 1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test - 1)

    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the neural network
    print("Testing Neural Network implementation...")

    # Dummy data for testing
    input_dim = 54
    batch_size = 32

    # Create model
    model = ForestCoverNet(input_dim=input_dim)
    advanced_model = ForestCoverNetAdvanced(input_dim=input_dim)

    print(
        f"Basic model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"Advanced model parameters: {sum(p.numel() for p in advanced_model.parameters()):,}")

    # Test forward pass
    dummy_input = torch.randn(batch_size, input_dim)
    output1 = model(dummy_input)
    output2 = advanced_model(dummy_input)

    print(f"Basic model output shape: {output1.shape}")
    print(f"Advanced model output shape: {output2.shape}")

    print("✅ Neural network models created successfully!")
