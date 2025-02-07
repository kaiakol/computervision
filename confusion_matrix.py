import glob
import os
import cv2
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
    return video_tensor

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
    
def load_model(model_path, device, num_classes=5):
    # Load the pretrained Swin3D model
    model = models.video.swin3d_b(weights=models.video.Swin3D_B_Weights.KINETICS400_V1)
    model.head = torch.nn.Linear(model.head.in_features, num_classes)

    # Load the model state_dict while mapping to CPU if necessary
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    return model


def get_confusion_matrix(model, data_loader, device):
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for videos, labels in tqdm(data_loader):
            videos, labels = videos.to(device), labels.to(device)
            videos = videos.squeeze(1)
            videos = videos.permute(0, 3, 1, 2, 4)

            outputs = model(videos)
            _, predicted = torch.max(outputs, 1)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    cm = confusion_matrix(true_labels, predicted_labels)
    return cm

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

def main(model_path, dataset_path, selected_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = load_model(model_path, device, num_classes=len(selected_classes))

    # Load the new dataset
    dataset = WorkoutVideoDataset(dataset_path, selected_classes=selected_classes)
    test_loader = DataLoader(dataset, batch_size=4, shuffle=False)

    # Get confusion matrix
    cm = get_confusion_matrix(model, test_loader, device)

    # Plot confusion matrix
    plot_confusion_matrix(cm, selected_classes)

if __name__ == "__main__":
    model_path = "final_swin3d_fitness.pth"  # Update with the path to your saved model
    dataset_path = "all_videos"        # Update with the path to your new dataset folder
    selected_classes = ["squat", "push-up", "pull Up", "russian twist", "plank"]  # Adjust as necessary

    main(model_path, dataset_path, selected_classes)
