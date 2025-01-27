import argparse
import os
from parse import parse_video, load_posesequence
from evaluate import evaluate_pose
import numpy as np

def load_training_data(labels_folder):
    """
    Load labeled keypoints data from the .npy files.

    Args:
        labels_folder (str): Path to the folder containing `correct.npy` and `incorrect.npy`.

    Returns:
        dict: A dictionary containing 'correct' and 'incorrect' PoseSequence lists.
    """
    correct_path = os.path.join(labels_folder, "correct.npy")
    incorrect_path = os.path.join(labels_folder, "incorrect.npy")
        
    correct_data = np.load(correct_path, allow_pickle=True)
    incorrect_data = np.load(incorrect_path, allow_pickle=True)

    # Load PoseSequence objects from the numpy arrays
    correct_sequences = [load_posesequence(sequence) for sequence in correct_data]
    incorrect_sequences = [load_posesequence(sequence) for sequence in incorrect_data]

    return {"correct": correct_sequences, "incorrect": incorrect_sequences}

def main():
    parser = argparse.ArgumentParser(description="Custom Pose Trainer with Mediapipe and KNN")
    parser.add_argument("--video", type=str, help="Path to video file for evaluation")
    parser.add_argument("--exercise", type=str, default="pushup", help="Exercise type: pushup, squat")

    args = parser.parse_args()

    training_folder = "data/" + args.exercise
    training_data = load_training_data(training_folder)

    if args.video:
        print("Processing video...")
        pose_seq = parse_video(args.video, "data/landmarks")
        correct, feedback = evaluate_pose(pose_seq, args.exercise, training_data)
        if correct:
            print("Exercise performed correctly!")
        else:
            print("Exercise could be improved:")
            for fb in feedback:
                print(f"- {fb}")
    else:
        print("No video file specified.")
        return

if __name__ == "__main__":
    main()
