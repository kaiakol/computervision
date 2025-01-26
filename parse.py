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

def main():
    parser = argparse.ArgumentParser(description='Pose Trainer Parser with Mediapipe')
    parser.add_argument('--input_folder', type=str, default='videos', help='Input folder containing video files')
    parser.add_argument('--output_folder', type=str, default='poses_compressed', help='Output folder for npy files')

    args = parser.parse_args()

    video_paths = glob.glob(os.path.join(args.input_folder, '*'))
    video_paths = sorted(video_paths)

    # Process all videos
    all_ps = []
    for video_path in video_paths:
        all_ps.append(parse_video(video_path, args.output_folder))

    return video_paths, all_ps

def parse_video(video_path, output_folder):
    """Parse a video file, extract Mediapipe pose landmarks, and save them as a numpy file.

    Args:
        video_path: path to the video file.
        output_folder: path to save the numpy array files of keypoints.
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

    # Convert keypoints to NumPy array
    keypoints_sequence = np.array(keypoints_sequence)

    # Save the keypoints
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(output_folder, video_name)
    os.makedirs(output_folder, exist_ok=True)
    np.save(output_dir, keypoints_sequence)

    return PoseSequence(keypoints_sequence)

def load_ps(filename):
    """Load a PoseSequence object from a given numpy file.

    Args:
        filename: file name of the numpy file containing keypoints.
    
    Returns:
        PoseSequence object with normalized joint keypoints.
    """
    all_keypoints = np.load(filename)
    return PoseSequence(all_keypoints)


if __name__ == '__main__':
    main()