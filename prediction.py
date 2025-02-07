import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the model
num_classes = 5
model = models.video.swin3d_b(weights=None)
model.head = nn.Linear(model.head.in_features, num_classes)
model.load_state_dict(torch.load("final_swin3d_fitness.pth", map_location=device))
model.to(device)
model.eval()

# Define class labels
class_labels = ["squat", "push-up", "pull Up", "russian twist", "plank"]

def preprocess_video(video_path, num_frames=16, frame_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, frame_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        frame = Image.fromarray(frame)
        frames.append(frame)

    cap.release()
    
    # Ensure exactly 16 frames
    if len(frames) < num_frames:
        frames = frames + [frames[-1]] * (num_frames - len(frames))  # Duplicate last frame if needed
    frames = frames[:num_frames]

    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to (C, H, W)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Apply transform to each frame
    video_tensor = torch.stack([transform(frame) for frame in frames])  # Shape (T, C, H, W)

    # Rearrange to (C, T, H, W) instead of (T, C, H, W)
    video_tensor = video_tensor.permute(1, 0, 2, 3)  # (C, T, H, W)

    # Add batch dimension to make it (B, C, T, H, W)
    video_tensor = video_tensor.unsqueeze(0)

    return video_tensor

def predict(video_path):
    video_tensor = preprocess_video(video_path).to(device).float()
    with torch.no_grad():
        output = model(video_tensor)
    _, predicted_class = torch.max(output, 1)
    predicted_action = class_labels[predicted_class.item()]
    actions_to_correct_format = {
        "squat": "squat",
        "push-up": "pushup",
        "pull Up": "pullup",
        "russian twist": "russiantwist",
        "plank": "plank"
    }
    action = actions_to_correct_format.get(predicted_action, predicted_action)
    print(f"Predicted action: {action}")
    return action 
    
