import os
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def evaluate_pose(pose_seq, exercise_type, training_data):
    """
    Evaluate a pose sequence for correctness using both geometric heuristics
    and a DTW-based binary nearest neighbor classifier.

    Args:
        pose_seq (PoseSequence): The pose sequence to evaluate.
        exercise_type (str): Type of exercise (e.g., 'pushup').
        training_data (dict): Dictionary containing training pose sequences for each exercise type.

    Returns:
        (bool, list): A tuple where the first element indicates if the exercise
                      is performed correctly overall, and the second element
                      contains a list of feedback from geometric and ML evaluations.
    """
    feedback = []

    # Geometric evaluation
    geometric_correct, geometric_feedback = geometric_evaluation(pose_seq, exercise_type)
    feedback.extend(geometric_feedback)

    # Machine learning evaluation
    if exercise_type not in training_data:
        feedback.append(f"No training data available for exercise: {exercise_type}")
        return False, feedback

    ml_correct, ml_feedback = ml_evaluation(pose_seq, training_data[exercise_type], k=3)
    feedback.extend(ml_feedback)

    # Combine geometric and ML evaluations, both must be correct
    overall_correct = geometric_correct and ml_correct
    if overall_correct:
        return True, ["Exercise performed correctly!"]
    return False, feedback


def geometric_evaluation(pose_seq, exercise_type):
    """
    Perform geometric evaluation based on joint angles and exercise-specific rules.

    Args:
        pose_seq (PoseSequence): The pose sequence to evaluate.
        exercise_type (str): Type of exercise (e.g., 'pushup', 'squat').

    Returns:
        (bool, list): A tuple where the first element indicates correctness,
                      and the second element contains feedback.
    """
    feedback = []
    correct = True

    if exercise_type == "pushup":
        # Forslag fra chatGPT
        for pose in pose_seq.poses:
            upper_arm = np.array([pose.rshoulder.x, pose.rshoulder.y]) - np.array([pose.relbow.x, pose.relbow.y])
            forearm = np.array([pose.relbow.x, pose.relbow.y]) - np.array([pose.rwrist.x, pose.rwrist.y])
            angle = np.degrees(np.arccos(np.dot(upper_arm, forearm) / (np.linalg.norm(upper_arm) * np.linalg.norm(forearm))))

            if angle > 70:
                correct = False
                feedback.append("Incomplete pushup: Fully extend your arms.")

    elif exercise_type == "squat":
        # Forslag fra chatGPT
        for pose in pose_seq.poses:
            hip_height = pose.rhip.y
            knee_height = pose.rknee.y
            ankle_height = pose.rankle.y

            if hip_height > knee_height or hip_height - ankle_height > 0.4:
                correct = False
                feedback.append("Incomplete squat: Lower your hips below your knees.")

    else:
        feedback.append(f"Geometric rules for {exercise_type} are not implemented.")

    return correct, feedback


def ml_evaluation(pose_seq, training_data, k=3):
    """
    Perform machine learning evaluation using a DTW-based nearest neighbor classifier.

    Args:
        pose_seq (PoseSequence): The pose sequence to evaluate.
        training_data (dict): Dictionary containing "correct" and "incorrect" training sequences.
        k (int): Number of nearest neighbors to consider.

    Returns:
        (bool, list): A tuple where the first element indicates correctness,
                      and the second element contains feedback.
    """
    feedback = []

    # Flatten the input pose sequence into a 2D array (frames x features)
    input_sequence = np.array([[part.x, part.y] for pose in pose_seq.poses for _, part in pose if part.exists])

    distances = []

    # Compare the input sequence against all training sequences
    for label, sequences in [("correct", training_data["correct"]), ("incorrect", training_data["incorrect"])]:
        for seq in sequences:
            train_sequence = np.array([[part.x, part.y] for pose in seq.poses for _, part in pose if part.exists])
            distance, _ = fastdtw(input_sequence, train_sequence, dist=euclidean)
            distances.append((distance, label))

    # Sort by distance and select the top k nearest neighbors
    distances = sorted(distances, key=lambda x: x[0])
    nearest_neighbors = distances[:k]

    # Count labels in the top k neighbors
    label_counts = {"correct": 0, "incorrect": 0}
    for _, label in nearest_neighbors:
        label_counts[label] += 1

    # Predict the label with the highest count
    predicted_label = max(label_counts, key=label_counts.get)

    if predicted_label == "incorrect":
        feedback.append("Machine learning evaluation suggests the form needs improvement.")
        return False, feedback

    return True, feedback
