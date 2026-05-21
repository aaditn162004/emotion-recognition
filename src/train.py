# src/train.py

import torch
import torch.optim as optim
import torch.nn as nn
from data_loader import get_data_loaders 
from model import EmotionCNN

def train_model():
    """Main function to train the emotion recognition model."""
    
    # --- 1. Hyperparameters and Setup ---
    NUM_EPOCHS = 50
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    DATA_PATH = '../data/fer2013' # Path to the folder containing train/ and test/
    MODEL_SAVE_PATH = '../saved_models/emotion_model.pth'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Load Data ---
    print("Loading data...")
    train_loader, test_loader, class_names = get_data_loaders(DATA_PATH, BATCH_SIZE)
    print(f"Loaded {len(train_loader.dataset)} training images.")

    # --- 3. Initialize Model, Loss, and Optimizer ---
    model = EmotionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 4. Training Loop ---
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train() # Set the model to training mode
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}")

    # --- 5. Save the Model ---
    print("Finished training.")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train_model()