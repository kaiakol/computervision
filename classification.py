import os
import glob
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from tqdm import tqdm
from PIL import Image

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# 1. Preprocess Video Data
# ============================
def preprocess_video(video_path, num_frames=16, frame_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, frame_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frames.append(frame)

    cap.release()
    
    # Ensure exactly 16 frames
    if len(frames) < num_frames:
        frames = frames + [frames[-1]] * (num_frames - len(frames))
    frames = frames[:num_frames]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Transform and stack the frames
    video_tensor = torch.stack([transform(frame) for frame in frames])
    
    # Correct the tensor shape to be (T, C, H, W)
    video_tensor = video_tensor.permute(0, 3, 1, 2)
    video_tensor = video_tensor.unsqueeze(0)
    
    return video_tensor

# ============================
# 2. Define Custom Dataset
# ============================
class WorkoutVideoDataset(Dataset):
    def __init__(self, root_dir, selected_classes=None, transform=None):
        self.root_dir = root_dir
        self.selected_classes = selected_classes if selected_classes else sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.selected_classes)}
        self.video_paths = []

        # Supported video formats
        video_formats = ['*.mp4', '*.MOV', '*.avi']

        for cls in self.selected_classes:
            class_path = os.path.join(root_dir, cls)
            for video_format in video_formats:
                video_files = glob.glob(os.path.join(class_path, video_format))
                self.video_paths.extend([(vf, self.class_to_idx[cls]) for vf in video_files])

        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path, label = self.video_paths[idx]
        video_tensor = preprocess_video(video_path)
        return video_tensor, torch.tensor(label)

# ============================
# 3. Load Pretrained Swin3D-B Model
# ============================
num_classes = 5
model = models.video.swin3d_b(weights=models.video.Swin3D_B_Weights.KINETICS400_V1)
model.head = nn.Linear(model.head.in_features, num_classes)
model = model.to(device)

# ============================
# 4. Training Setup
# ============================
selected_classes = ["squat", "push-up", "pull Up", "russian twist", "plank"]

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),  # Rotate randomly by 10 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Random color jitter
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = WorkoutVideoDataset("workout_data", selected_classes=selected_classes, transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoader for training and testing
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# ============================
# 5. Training Loop
# ============================
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

# Early stopping parameters
early_stopping_patience = 3
best_val_loss = float("inf")
patience_counter = 0

num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for videos, labels in tqdm(dataloader):
        videos, labels = videos.to(device), labels.to(device)
        videos = videos.squeeze(1)
        videos = videos.permute(0, 3, 1, 2, 4)

        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    # Compute epoch loss and accuracy
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total * 100
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    # Validation loop (testing the model on the test set)
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for videos, labels in tqdm(test_loader):
            videos, labels = videos.to(device), labels.to(device)
            videos = videos.squeeze(1)
            videos = videos.permute(0, 3, 1, 2, 4)

            outputs = model(videos)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)

    test_epoch_loss = test_loss / len(test_loader)
    test_epoch_acc = test_correct / test_total * 100
    test_losses.append(test_epoch_loss)
    test_accuracies.append(test_epoch_acc)

    print(f"Test Loss: {test_epoch_loss:.4f}, Test Accuracy: {test_epoch_acc:.2f}%")

    # Early stopping logic
    if test_epoch_loss < best_val_loss:
        best_val_loss = test_epoch_loss
        patience_counter = 0  # Reset patience counter
        torch.save(model.state_dict(), "/content/drive/MyDrive/best_swin3d_fitness.pth")  # Save best model
    else:
        patience_counter += 1
        print(f"No improvement for {patience_counter} epochs. Best loss: {best_val_loss:.4f}")

    if patience_counter >= early_stopping_patience:
        print("Early stopping triggered! Stopping training.")
        break

# Save training and test logs
df = pd.DataFrame({
    "Epoch": range(1, len(train_losses) + 1),
    "Train Loss": train_losses,
    "Train Accuracy": train_accuracies,
    "Test Loss": test_losses,
    "Test Accuracy": test_accuracies
})
df.to_csv("training_log.csv", index=False)
print("Training log saved!")

torch.save(model.state_dict(), "final_swin3d_fitness.pth")
print("Final model saved!")
