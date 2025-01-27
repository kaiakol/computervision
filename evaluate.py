import os
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import json

def evaluate_pose(pose_seq, exercise_type, training_data):
    """
    Evaluate a pose sequence for correctness using both geometric heuristics
    and a DTW-based binary nearest neighbor classifier.

    Args:
        pose_seq (PoseSequence): The pose sequence to evaluate.
        exercise_type (str): Type of exercise.
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
    knn_correct, knn_feedback = knn_evaluation(pose_seq, exercise_type, training_data, k=3)
    feedback.extend(knn_feedback)

    # Combine geometric and ML evaluations, both must be correct
    overall_correct = geometric_correct and knn_correct
    if overall_correct:
        return True, ["Exercise performed correctly!"]
    return False, feedback

def geometric_evaluation(pose_seq, exercise_type, base_dir="data"):
    """
    Perform geometric evaluation based on joint angles and exercise-specific rules.

    Args:
        pose_seq (PoseSequence): The pose sequence to evaluate.
        exercise_type (str): Type of exercise (e.g., 'pushup', 'squat').
        base_dir (str): Base directory containing exercise folders.

    Returns:
        (bool, list): A tuple where the first element indicates correctness,
                      and the second element contains feedback.
    """
    try:
        rules = load_exercise_rules(exercise_type, base_dir)
        validate_exercise_rules(rules)
    except (FileNotFoundError, ValueError) as e:
        return (False, [str(e)])

    return evaluate_exercise(pose_seq, rules)


def evaluate_exercise(pose_seq, rules):
    """
    Evaluate a PoseSequence using the provided rules.

    Args:
        pose_seq (PoseSequence): The pose sequence to evaluate.
        rules (dict): The rules loaded from JSON.

    Returns:
        (bool, list): A tuple where the first element indicates correctness,
                      and the second element contains feedback.
    """
    feedback = []
    correct = True

    for angle_rule in rules["angle_rules"]:
        joints = angle_rule["joints"]
        min_angle = angle_rule.get("min_angle")
        max_angle = angle_rule.get("max_angle")
        feedback_msg = angle_rule["feedback"]

        angles = compute_joint_angles(pose_seq, joints)

        # Check angle constraints
        if min_angle is not None and any(angle < min_angle for angle in angles):
            correct = False
            feedback.append(feedback_msg)

        if max_angle is not None and any(angle > max_angle for angle in angles):
            correct = False
            feedback.append(feedback_msg)

    if correct:
        feedback.append("Exercise performed correctly!")

    return correct, feedback

import numpy as np

def compute_joint_angles(pose_seq, joints):
    """
    Compute angles between joints in a PoseSequence.

    Args:
        pose_seq (PoseSequence): The sequence of poses to evaluate.
        joints (list): A list of joint names that define the angle (e.g., ["lshoulder", "lelbow", "lwrist"]).

    Returns:
        list: A list of angles (in degrees) for each pose in the sequence.
    """
    if len(joints) != 3:
        raise ValueError("Exactly 3 joints must be provided to compute an angle.")

    joint1, joint2, joint3 = joints
    angles = []

    for pose in pose_seq.poses:
        # Get the joint positions
        part1 = getattr(pose, joint1)
        part2 = getattr(pose, joint2)
        part3 = getattr(pose, joint3)

        # Ensure all joints exist in the current pose
        if not (part1.exists and part2.exists and part3.exists):
            continue

        # Compute vectors
        vec1 = np.array([part1.x - part2.x, part1.y - part2.y])
        vec2 = np.array([part3.x - part2.x, part3.y - part2.y])

        # Normalize vectors
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)

        # Compute angle between vectors using dot product
        dot_product = np.dot(vec1_norm, vec2_norm)
        angle = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))

        angles.append(angle)

    return angles

def knn_evaluation(pose_seq, exercise_type, training_data, k=3):
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

    if exercise_type not in training_data:
        feedback = "No training data available for exercise: {exercise_type}"
        return False, feedback

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
        feedback = "Incorrect form detected."
        return False, feedback

    feedback = "Correct form detected."
    return True, feedback


def load_exercise_rules(exercise_type, base_dir="data"):
    """
    Load exercise-specific rules from a JSON file in the exercise folder.

    Args:
        exercise_type (str): The name of the exercise (e.g., 'pushup').
        base_dir (str): The base directory containing exercise folders.

    Returns:
        dict: The rules dictionary for the exercise.

    Raises:
        FileNotFoundError: If the rules file is not found.
        ValueError: If the JSON file is invalid or improperly formatted.
    """
    file_path = os.path.join(base_dir, exercise_type, "rules.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Rules file not found for exercise '{exercise_type}' at {file_path}.")
    
    try:
        with open(file_path, "r") as f:
            rules = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse rules.json for '{exercise_type}': {str(e)}")
    
    return rules

def validate_exercise_rules(rules):
    """
    Validate the structure of the exercise rules JSON.

    Args:
        rules (dict): The rules dictionary.

    Raises:
        ValueError: If the rules dictionary is invalid.
    """
    required_keys = ["keypoints", "angle_rules"]

    for key in required_keys:
        if key not in rules:
            raise ValueError(f"Missing required key '{key}' in rules.")

    if not isinstance(rules["keypoints"], list):
        raise ValueError("'keypoints' should be a list.")

    if not isinstance(rules["angle_rules"], list):
        raise ValueError("'angle_rules' should be a list.")

    for angle_rule in rules["angle_rules"]:
        if not all(k in angle_rule for k in ["joints", "min_angle", "max_angle", "feedback"]):
            raise ValueError("Each angle rule must contain 'joints', 'min_angle', 'max_angle', and 'feedback'.")
