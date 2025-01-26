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
        exercise_type (str): Type of exercise (e.g., 'bicep_curl').
        training_data (dict): Dictionary containing training pose sequences for each exercise type.

    Returns:
        (bool, list): A tuple where the first element indicates if the exercise
                      is performed correctly overall, and the second element
                      contains a list of feedback from geometric and ML evaluations.
    """

    feedback = []

    # Geometric evaluation
    geometric_correct = True
    if exercise_type == "pushup":
        correct, feedback = evaluate_pushup(pose_seq)
        """
        # Example geometric rule: Check elbow and shoulder angles
        for pose in pose_seq.poses:
            upper_arm = np.array([pose.rshoulder.x, pose.rshoulder.y]) - np.array([pose.relbow.x, pose.relbow.y])
            forearm = np.array([pose.relbow.x, pose.relbow.y]) - np.array([pose.rwrist.x, pose.rwrist.y])
            angle = np.degrees(np.arccos(np.dot(upper_arm, forearm) / (np.linalg.norm(upper_arm) * np.linalg.norm(forearm))))

            if angle > 70:  # Example threshold for incomplete curls
                geometric_correct = False
                feedback.append("Your bicep curl is incomplete. Fully contract your arm.")
        """
    elif exercise_type == "squat":
        correct, feedback = evaluate_squat(pose_seq)

    # Machine learning evaluation
    if exercise_type not in training_data:
        feedback.append(f"No training data available for exercise: {exercise_type}")
        return False, feedback

    ml_correct = True
    input_sequence = np.array([part.x for pose in pose_seq.poses for _, part in pose])

    distances = []
    for label, sequences in [("correct", training_data[exercise_type]["correct"]),
                             ("incorrect", training_data[exercise_type]["incorrect"])]:
        for seq in sequences:
            seq_flattened = np.array([part.x for pose in seq.poses for _, part in pose])
            distance, _ = fastdtw(input_sequence, seq_flattened, dist=euclidean)
            distances.append((distance, label))

    closest_match = min(distances, key=lambda x: x[0])
    predicted_label = closest_match[1]

    if predicted_label == "incorrect":
        ml_correct = False
        feedback.append("Machine learning evaluation suggests the form needs improvement.")

    # Combine geometric and ML evaluations, both must be correct
    overall_correct = geometric_correct and ml_correct
    if overall_correct:
        return True, ["Exercise performed correctly!"]
    return False, feedback


def evaluate_pushup(pose_seq):
    pass #TODO

def evaluate_squat(pose_seq):
    pass #TODO