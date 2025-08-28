# train_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

class DeepFakeDetector(nn.Module):
    """Simplified CNN model for DeepFake detection using PyTorch"""
    def __init__(self):
        super(DeepFakeDetector, self).__init__()
        
        # Simplified CNN architecture
        self.features = nn.Sequential(
            # Input: (batch_size, 3, 10, 128, 128)
            nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 10 * 16 * 16, 512),  # Adjusted based on pooling
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class VideoDataset(Dataset):
    """Custom dataset for video frames"""
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # Convert to tensor and adjust dimensions for PyTorch
        # PyTorch expects (C, T, H, W) but we have (T, H, W, C)
        frames = self.X[idx]
        frames = np.transpose(frames, (3, 0, 1, 2))  # (C, T, H, W)
        return torch.FloatTensor(frames), torch.FloatTensor([self.y[idx]])

def create_synthetic_dataset(num_samples=200, num_frames=10, img_size=128):
    """Create synthetic dataset for demonstration"""
    print("Creating synthetic dataset...")
    
    # Real videos - more consistent frames
    X_real = np.random.rand(num_samples, num_frames, img_size, img_size, 3) * 0.3 + 0.4
    
    # Fake videos - more variation between frames
    X_fake = []
    for _ in range(num_samples):
        base = np.random.rand(img_size, img_size, 3) * 0.5 + 0.2
        video = []
        for __ in range(num_frames):
            variation = np.random.randn(img_size, img_size, 3) * 0.1
            frame = np.clip(base + variation, 0, 1)
            video.append(frame)
        X_fake.append(np.array(video))
    X_fake = np.array(X_fake)
    
    X = np.concatenate([X_real, X_fake])
    y = np.concatenate([np.zeros(num_samples), np.ones(num_samples)])
    
    return X, y

def train_model():
    """Train the DeepFake detection model"""
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Create synthetic data
    X, y = create_synthetic_dataset(num_samples=100, num_frames=10, img_size=128)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create datasets and data loaders
    train_dataset = VideoDataset(X_train, y_train)
    test_dataset = VideoDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # Create model, loss function, and optimizer
    model = DeepFakeDetector().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 10
    train_losses = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}')
        
        # Calculate average training loss
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                predicted = (output > 0.5).float()
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        
        print(f'Epoch: {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Test Accuracy: {accuracy:.2f}%')
    
    # Save model
    torch.save(model.state_dict(), 'models/deepfake_model.pth')
    print("Model saved to models/deepfake_model.pth")
    
    # Save training history
    training_history = {
        'train_losses': train_losses,
        'test_accuracies': test_accuracies
    }
    joblib.dump(training_history, 'models/training_history.pkl')
    
    return model, accuracy / 100

if __name__ == "__main__":
    # Train the model
    model, accuracy = train_model()
    print(f"Model trained with accuracy: {accuracy:.2%}")