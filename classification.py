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
    video_tensor = video_tensor.permute(0, 3, 1, 2)  # Change shape to (T, C, H, W)
    video_tensor = video_tensor.unsqueeze(0)
    print(video_tensor.shape)  # Debugging: Check the shape of the output tensor
    
    return video_tensor


# ============================
# 2. Define Custom Dataset
# ============================
class WorkoutVideoDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.video_paths = []

        # Supported video formats
        video_formats = ['*.mp4', '*.MOV', '*.avi']

        for cls in self.classes:
            class_path = os.path.join(root_dir, cls)
            for video_format in video_formats:
                video_files = glob.glob(os.path.join(class_path, video_format))
                self.video_paths.extend([(vf, self.class_to_idx[cls]) for vf in video_files])

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path, label = self.video_paths[idx]
        video_tensor = preprocess_video(video_path)
        return video_tensor, torch.tensor(label)

# ============================
# 3. Load Pretrained Swin3D-T Model
# ============================
num_classes = 22  # Adjust based on dataset
model = models.video.swin3d_t(weights=models.video.Swin3D_T_Weights.KINETICS400_V1)
model.head = nn.Linear(model.head.in_features, num_classes)
model = model.to(device)

# ============================
# 4. Training Setup
# ============================
dataset = WorkoutVideoDataset("workout_data")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# ============================
# 5. Training Loop
# ============================
train_losses = []
train_accuracies = []

# Early stopping parameters
early_stopping_patience = 3  # Stop after 'n' epochs with no improvement
best_val_loss = float("inf")
patience_counter = 0

num_epochs = 20  # Increase epochs but rely on early stopping

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
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total * 100
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    # Early stopping logic
    if epoch_loss < best_val_loss:
        best_val_loss = epoch_loss
        patience_counter = 0  # Reset patience counter
        torch.save(model.state_dict(), "best_swin3d_fitness.pth")  # Save best model
    else:
        patience_counter += 1
        print(f"No improvement for {patience_counter} epochs. Best loss: {best_val_loss:.4f}")

    if patience_counter >= early_stopping_patience:
        print("Early stopping triggered! Stopping training.")
        break

# Save training log
df = pd.DataFrame({"Epoch": range(1, len(train_losses) + 1), "Loss": train_losses, "Accuracy": train_accuracies})
df.to_csv("training_log.csv", index=False)
print("Training log saved!")

# Save final model
torch.save(model.state_dict(), "final_swin3d_fitness.pth")
print("Final model saved!")
# ============================
# 6. Inference Function
# ============================
def predict(video_path):
    model.eval()
    video_tensor = preprocess_video(video_path).to(device).float()
    with torch.no_grad():
        output = model(video_tensor)
    top5_probs, top5_classes = torch.topk(torch.nn.functional.softmax(output, dim=1), 5)
    class_labels = dataset.classes  # Get class names
    print("Predictions:")
    for i in range(5):
        action = class_labels[top5_classes[0, i].item()]
        probability = top5_probs[0, i].item() * 100
        print(f"{action}: {probability:.2f}%")

# ============================
# 7. Run Inference
# ============================
test_video = "workout_data/squat/squat_1.MOV"
predict(test_video)
