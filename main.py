import argparse
from parse import parse_video
from evaluate import evaluate_pose
from evaluate import load_training_data
from prediction import predict

def main():
    parser = argparse.ArgumentParser(description="Custom Pose Trainer with Mediapipe and KNN")
    parser.add_argument("--video", type=str, required=True, help="Path to video file for evaluation")
    parser.add_argument("--exercise", type=str, help="Exercise type: pushup, squat, etc. (Optional, will be predicted if not provided)")

    args = parser.parse_args()

    if not args.video:
        print("No video file specified.")
        return

    if not args.exercise:
        print("No exercise type provided. Predicting exercise type...")
        args.exercise = predict(args.video)

    pose_seq = parse_video(args.video, "data/landmarks")
    training_data = load_training_data(args.exercise)
    evaluate_pose(pose_seq, args.exercise, training_data)

if __name__ == "__main__":
    main()
