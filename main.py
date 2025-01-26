"""Pose trainer main script using Mediapipe."""

import argparse
import os
import numpy as np
from parse import parse_video, load_ps
from evaluate import evaluate_pose
from utils import load_training_data


def main():
    parser = argparse.ArgumentParser(description='Pose Trainer with Mediapipe and ML')
    parser.add_argument('--mode', type=str, default='evaluate', help='Pose Trainer application mode.\n'
                        'One of evaluate, batch_process, evaluate_npy.')
    parser.add_argument('--input_folder', type=str, default='videos', help='Input folder for videos.')
    parser.add_argument('--output_folder', type=str, default='poses_compressed', help='Output folder for poses.')
    parser.add_argument('--video', type=str, help='Input video filepath for evaluation.')
    parser.add_argument('--file', type=str, help='Input .npy file for evaluation.')
    parser.add_argument('--exercise', type=str, default='bicep_curl', help='Exercise type to evaluate.')
    parser.add_argument('--training_folder', type=str, default='training_data', help='Folder containing training data.')

    args = parser.parse_args()

    # Load training data
    training_data = load_training_data(args.training_folder)

    if args.mode == 'evaluate':
        if args.video:
            print('Processing video file...')
            pose_seq = parse_video(args.video, args.output_folder)
            correct, feedback = evaluate_pose(pose_seq, args.exercise, training_data)
            if correct:
                print('Exercise performed correctly!')
            else:
                print('Exercise could be improved:')
                for fb in feedback:
                    print(f"- {fb}")
        else:
            print('No video file specified.')
            return

    elif args.mode == 'evaluate_npy':
        if args.file:
            pose_seq = load_ps(args.file)
            correct, feedback = evaluate_pose(pose_seq, args.exercise, training_data)
            if correct:
                print('Exercise performed correctly!')
            else:
                print('Exercise could be improved:')
                for fb in feedback:
                    print(f"- {fb}")
        else:
            print('No .npy file specified.')
            return

    else:
        print('Unrecognized mode option.')
        return

if __name__ == "__main__":
    main()
