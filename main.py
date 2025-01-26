import argparse
import os
from parse import parse_video, load_ps, process_video_dataset
from evaluate import evaluate_pose
import numpy as np

def load_training_data(labels_folder):
    """
    Load labeled keypoints data from the .npy files.

    Args:
        labels_folder (str): Path to the folder containing `correct.npy` and `wrong.npy`.

    Returns:
        dict: A dictionary containing 'correct' and 'wrong' PoseSequence lists.
    """
    correct_path = os.path.join(labels_folder, "correct.npy")
    wrong_path = os.path.join(labels_folder, "wrong.npy")

    correct_sequences = [load_ps(sequence) for sequence in np.load(correct_path, allow_pickle=True)]
    wrong_sequences = [load_ps(sequence) for sequence in np.load(wrong_path, allow_pickle=True)]

    return {"correct": correct_sequences, "wrong": wrong_sequences}

def main():
    parser = argparse.ArgumentParser(description="Pose Trainer with Mediapipe and ML")
    parser.add_argument("--mode", type=str, default="evaluate", help="Application mode: evaluate, batch_process, or evaluate_npy")
    parser.add_argument("--input_folder", type=str, default="videos", help="Folder containing input videos")
    parser.add_argument("--output_folder", type=str, default="poses_compressed", help="Folder to save processed keypoints")
    parser.add_argument("--video", type=str, help="Input video file for evaluation")
    parser.add_argument("--file", type=str, help="Input .npy file for evaluation")
    parser.add_argument("--exercise", type=str, default="pushup", help="Exercise type (e.g., pushup)")
    parser.add_argument("--training_folder", type=str, default="dataset/labels", help="Folder containing training data")

    args = parser.parse_args()

    # Load training data
    training_data = load_training_data(args.training_folder)

    if args.mode == "evaluate":
        if args.video:
            print("Processing video...")
            pose_seq = parse_video(args.video, args.output_folder)
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

    elif args.mode == "batch_process":
        process_video_dataset(args.input_folder, args.output_folder)

    elif args.mode == "evaluate_npy":
        if args.file:
            pose_seq = load_ps(args.file)
            correct, feedback = evaluate_pose(pose_seq, args.exercise, training_data)
            if correct:
                print("Exercise performed correctly!")
            else:
                print("Exercise could be improved:")
                for fb in feedback:
                    print(f"- {fb}")
        else:
            print("No .npy file specified.")
            return

    else:
        print("Unrecognized mode.")
        return

if __name__ == "__main__":
    main()
