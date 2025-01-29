import argparse
import numpy as np
import os
import cv2
import mediapipe as mp
from pose import PoseSequence

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
    pose_sequences = []

    # Initialize Mediapipe Pose
    mp_pose = mp.solutions.pose
    pose_estimator = mp_pose.Pose()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_estimator.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            keypoints = [
                [landmarks[i].x, landmarks[i].y, landmarks[i].visibility]
                for i in [
                    mp_pose.PoseLandmark.NOSE.value,
                    mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                    mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                    mp_pose.PoseLandmark.LEFT_ELBOW.value,
                    mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                    mp_pose.PoseLandmark.LEFT_WRIST.value,
                    mp_pose.PoseLandmark.RIGHT_WRIST.value,
                    mp_pose.PoseLandmark.LEFT_HIP.value,
                    mp_pose.PoseLandmark.RIGHT_HIP.value,
                    mp_pose.PoseLandmark.LEFT_KNEE.value,
                    mp_pose.PoseLandmark.RIGHT_KNEE.value,
                    mp_pose.PoseLandmark.LEFT_ANKLE.value,
                    mp_pose.PoseLandmark.RIGHT_ANKLE.value,
                    mp_pose.PoseLandmark.LEFT_HEEL.value,
                    mp_pose.PoseLandmark.RIGHT_HEEL.value,
                    mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value,
                    mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value
                ]
            ]
            
            pose_sequences.append(keypoints)

    cap.release()

    # Save the pose sequences as a numpy file
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(output_folder, video_name)
    os.makedirs(output_folder, exist_ok=True)
    np.save(output_dir, pose_sequences)

    # Return a PoseSequence object created from the extracted keypoints
    return PoseSequence(pose_sequences)

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

# Define the Mediapipe indices corresponding to the 17 keypoints you are using
MEDIAPIPE_LANDMARKS = [
    0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
]

def load_posesequence(file_path):
    """
    Load a PoseSequence from a .npy file containing a (num_frames, 33, 3) array.
    
    Args:
        file_path (str): Path to the NumPy file.
    
    Returns:
        PoseSequence: The initialized pose sequence.
    """
    data = np.load(file_path)
    num_frames = data.shape[0]

    sequence = []
    for frame in range(num_frames):
        frame_data = data[frame]  # Get landmarks for this frame
        selected_parts = [frame_data[i].tolist() for i in MEDIAPIPE_LANDMARKS]
        sequence.append(selected_parts)

    return PoseSequence(sequence)

def main():
    """
    Main function to parse videos in a folder and save their pose keypoints as .npy files.
    """
    parser = argparse.ArgumentParser(description='Pose Trainer Parser with Mediapipe')
    parser.add_argument('--input_folder', type=str, help='Input folder containing video files')
    parser.add_argument('--output_folder', type=str, help='Output folder for npy files')

    args = parser.parse_args()
    process_video_dataset(args.input_folder, args.output_folder)

if __name__ == '__main__':
    main()