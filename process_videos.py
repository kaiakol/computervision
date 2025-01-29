import os
import cv2
import numpy as np
import mediapipe as mp

def extract_landmarks_from_video(video_path):
    """
    Process a video and extract pose landmarks (x, y, visibility) from each frame.

    Args:
        video_path (str): Path to the video file.

    Returns:
        list: A list of 33 landmarks (each landmark is a tuple of (x, y, visibility)) for each frame.
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    landmarks_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            frame_landmarks = []
            for landmark in results.pose_landmarks.landmark:
                frame_landmarks.append([landmark.x, landmark.y, landmark.visibility])
            landmarks_list.append(frame_landmarks)
        else:
            landmarks_list.append(np.zeros((33, 3)))  # 33 landmarks, each with 3 values (x, y, visibility)

    cap.release()
    return landmarks_list

def process_videos_in_folder(videos_folder, output_file):
    """
    Process all video files in a folder and save all landmarks (x, y, visibility) into a single .npy file.

    Args:
        videos_folder (str): Path to the folder containing video files.
        output_file (str): Path to the output .npy file where the landmarks will be saved.
    """
    all_landmarks = []

    for filename in os.listdir(videos_folder):
        if filename.endswith(".mp4") or filename.endswith(".avi"):  # Add other video formats if needed
            video_path = os.path.join(videos_folder, filename)
            print(f"Processing video: {video_path}")
            landmarks = extract_landmarks_from_video(video_path)
            all_landmarks.extend(landmarks)

    # Save the landmarks to a single .npy file
    np.save(output_file, np.array(all_landmarks))
    print(f"Landmarks saved to {output_file}")

if __name__ == "__main__":
    videos_folder = "data/pushup/incorrect"  # Replace with the folder path
    output_file = "data/pushup/incorrect2.npy"  # Path to save the output landmarks file

    process_videos_in_folder(videos_folder, output_file)

