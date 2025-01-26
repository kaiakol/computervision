import argparse
import glob
import numpy as np
import os
import cv2
import mediapipe as mp
from pose import PoseSequence

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose_estimator = mp_pose.Pose()

def parse_video(video_path, output_folder):
    """
    Parse a video file, extract Mediapipe pose landmarks, and save them as a numpy file.

    Args:
        video_path: Path to the video file.
        output_folder: Path to save the numpy array files of keypoints.

    Returns:
        PoseSequence: A PoseSequence object containing the extracted keypoints.
    """
    cap = cv2.VideoCapture(video_path)
    keypoints_sequence = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB for Mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_estimator.process(frame_rgb)

        # Extract keypoints
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            keypoints = np.array([
                [lm.x, lm.y, lm.visibility]
                for lm in landmarks
            ])
            keypoints_sequence.append(keypoints)
        else:
            # Append zeros for missing frames
            keypoints_sequence.append(np.zeros((33, 3)))  # Mediapipe has 33 keypoints

    cap.release()

    keypoints_sequence = np.array(keypoints_sequence)

    # Save the keypoints
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(output_folder, video_name)
    os.makedirs(output_folder, exist_ok=True)
    np.save(output_dir, keypoints_sequence)

    return PoseSequence(keypoints_sequence)

def process_video_dataset(input_folder, output_folder):
    """
    Process all videos in the input folder using Mediapipe and save extracted keypoints.

    Args:
        input_folder: Path to the folder containing video files.
        output_folder: Path to save processed keypoints.
    """
    video_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi'))]

    for video_path in video_paths:
        print(f"Processing: {video_path}")
        parse_video(video_path, output_folder)

def load_ps(filename):
    """
    Load a PoseSequence object from a .npy file.
    """
    all_keypoints = np.load(filename)
    return PoseSequence(all_keypoints)


def main():
    """
    Main function to parse videos in a folder and save their pose keypoints as .npy files.
    """
    parser = argparse.ArgumentParser(description='Pose Trainer Parser with Mediapipe')
    parser.add_argument('--input_folder', type=str, default='videos', help='Input folder containing video files')
    parser.add_argument('--output_folder', type=str, default='poses_compressed', help='Output folder for npy files')

    args = parser.parse_args()

    # Process videos in the input folder and save results to the output folder
    process_video_dataset(args.input_folder, args.output_folder)

if __name__ == '__main__':
    main()