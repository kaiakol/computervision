import argparse
from parse import parse_video
from evaluate import evaluate_pose
from evaluate import load_training_data

def main():
    parser = argparse.ArgumentParser(description="Custom Pose Trainer with Mediapipe and KNN")
    parser.add_argument("--video", type=str, help="Path to video file for evaluation")
    parser.add_argument("--exercise", type=str, default="pushup", help="Exercise type: pushup, squat")

    args = parser.parse_args()

    if args.video:
        #print("Processing video...")
        pose_seq = parse_video(args.video, "data/landmarks")
        training_data = load_training_data(args.exercise)
        evaluate_pose(pose_seq, args.exercise, training_data)
    else:
        print("No video file specified.")
        return

if __name__ == "__main__":
    main()
