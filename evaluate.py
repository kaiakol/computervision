import os
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import json

from parse import load_posesequence

def evaluate_pose(pose_seq, exercise_type, training_data):
    """
    Evaluate a pose sequence for correctness using both geometric heuristics
    and a DTW-based binary nearest neighbor classifier.

    Args:
        pose_seq (PoseSequence): The pose sequence to evaluate.
        exercise_type (str): Type of exercise.
        training_data (dict): Dictionary containing "correct" and "incorrect" training sequences.
    """
    # Geometric evaluation
    print("Performing geometric evaluation...")
    geometric_correct, geometric_feedback = geometric_evaluation(pose_seq, exercise_type)
    if geometric_correct:
        print("Geometric evaluation: Correct form")
    else: 
        print(f"Geometric evaluation: Incorrect form. {geometric_feedback}")

    # Machine learning evaluation
    print("Performing KNN evaluation...")
    knn_correct =  knn_evaluation(pose_seq, exercise_type, training_data, k=3)
    if knn_correct:
        print("KNN evaluation: Correct form")
    else:
        print(f"KNN evaluation: Incorrect form")
    
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
        #validate_exercise_rules(rules)
    except (FileNotFoundError, ValueError) as e:
        return False, [str(e)]

    # Detect perspective and normalize if necessary
    perspective = pose_seq.perspective
    if perspective == "left":
        pose_seq.normalize_perspective()  # Normalize to the right-side view

    # Compute joint angle statistics for all relevant joints
    for angle_rule in rules["angle_rules"]:
        # Use side-specific joint names based on perspective
        joints = angle_rule["sides"][perspective] if perspective in ["left", "right"] else angle_rule["sides"]["front"]

        if len(joints) == 3:
            pose_seq.compute_joint_angle_statistics(joints)
            # print(pose_seq.min_angles)
            # print(pose_seq.max_angles)

        #else:
        #    print(f"Computing motion range for {joints}")
        #    pose_seq.compute_motion_ranges(joints)

    # Evaluate the exercise using the provided rules
    return evaluate_exercise(pose_seq, rules, perspective)

def evaluate_exercise(pose_seq, rules, perspective):
    """
    Evaluate a PoseSequence using the provided rules and perspective.

    Args:
        pose_seq (PoseSequence): The pose sequence to evaluate.
        rules (dict): The rules loaded from JSON.
        perspective (str): Detected perspective ("left", "right", "front").

    Returns:
        (bool, list): A tuple where the first element indicates correctness,
                      and the second element contains feedback.
    """
    feedback = []
    correct = True

    for angle_rule in rules["angle_rules"]:
        joints_left = angle_rule["sides"]["left"]
        joints_right = angle_rule["sides"]["right"]
        min_angle_lower = angle_rule.get("min_angle_lower")
        max_angle_upper = angle_rule.get("max_angle_upper")
        max_angle_lower = angle_rule.get("max_angle_lower")
        min_angle_upper = angle_rule.get("min_angle_upper")
        min_motion = angle_rule.get("min_motion")
        max_motion = angle_rule.get("max_motion")
        max_motion_front = angle_rule.get("max_motion_front")
        feedback_msg = angle_rule["feedback"]
        feedback_msg_motion = angle_rule["feedback_motion"]

        # Evaluate based on perspective
        if perspective in ["left", "right"]:
            joints = angle_rule["sides"][perspective]
            min_joint_angle = pose_seq.min_angles.get(tuple(joints))
            max_joint_angle = pose_seq.max_angles.get(tuple(joints))
            motion_range = pose_seq.motion_ranges.get(joints[0])

            # print(f"Min joint angle: {min_joint_angle:.2f}°" if min_joint_angle is not None else "None")
            # print(f"Max joint angle: {max_joint_angle:.2f}°" if max_joint_angle is not None else "None")
            # print(f"Motion range: {motion_range:.2f}" if motion_range is not None else "Motion range = None")

            if min_angle_lower is not None and min_angle_upper is not None and min_joint_angle is not None and (min_angle_upper < min_joint_angle or min_joint_angle < min_angle_lower):
                correct = False
                feedback.append(f"{feedback_msg} Minimum angle: {min_joint_angle:.2f}°. Minimum angle should be between: {min_angle_lower}° and {min_angle_upper}°")

            if max_angle_lower is not None and max_angle_upper is not None and max_joint_angle is not None and (max_angle_upper < max_joint_angle or max_joint_angle < max_angle_lower):
                correct = False
                feedback.append(f"{feedback_msg} Maximum angle: {max_joint_angle:.2f}°. Maximum angle should be between: {max_angle_lower}° and {max_angle_upper}°")

            if min_motion is not None and motion_range is not None and motion_range < min_motion:
                correct = False
                feedback.append(f"{feedback_msg_motion} Total movement: {motion_range:.2f}. Movement should be ≥{min_motion}")

            if max_motion is not None and motion_range is not None and motion_range > max_motion:
                correct = False
                feedback.append(f"{feedback_msg_motion} Total movement: {motion_range:.2f}. Movement should be ≤{max_motion}")

        elif perspective == "front":
            # Evaluate both left and right sides
            for joints in [joints_left, joints_right]:
                min_joint_angle = pose_seq.min_angles.get(tuple(joints))
                max_joint_angle = pose_seq.max_angles.get(tuple(joints))
                motion_range = pose_seq.motion_ranges.get(joints[0])

            if min_angle_lower is not None and min_angle_upper is not None and min_joint_angle is not None and (min_angle_upper < min_joint_angle or min_joint_angle < min_angle_lower):
                correct = False
                feedback.append(f"{feedback_msg} Minimum angle: {min_joint_angle:.2f}°. Minimum angle should be between: {min_angle_lower}° and {min_angle_upper}°")

            if max_angle_lower is not None and max_angle_upper is not None and max_joint_angle is not None and (max_angle_upper < max_joint_angle or max_joint_angle < max_angle_lower):
                correct = False
                feedback.append(f"{feedback_msg} Maximum angle: {max_joint_angle:.2f}°. Maximum angle should be between: {max_angle_lower}° and {max_angle_upper}°")

            if min_motion is not None and motion_range is not None and motion_range < min_motion:
                correct = False
                feedback.append(f"{feedback_msg_motion} Total movement: {motion_range:.2f}. Movement should be ≥ {min_motion}")

            if max_motion is not None and motion_range is not None and motion_range > max_motion:
                correct = False
                feedback.append(f"{feedback_msg_motion} Total movement: {motion_range:.2f}. Movement should be ≤ {max_motion}")

    return correct, feedback

def knn_evaluation(pose_seq, exercise_type, training_data, k=3):
    """
    Perform machine learning evaluation using a DTW-based nearest neighbor classifier.

    Args:
        pose_seq (PoseSequence): The pose sequence to evaluate.
        exercise_type (str): Type of exercise (e.g., 'pushup', 'squat').
        training_data (dict): Dictionary containing "correct" and "incorrect" training sequences.
        k (int): Number of nearest neighbors to consider.

    Returns:
        bool: Whether the exercise was performed correctly.
    """
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
        return False
    else: 
        return True

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

    # Check for required keys
    for key in required_keys:
        if key not in rules:
            raise ValueError(f"Missing required key '{key}' in rules.")

    # Validate the 'keypoints' list
    if not isinstance(rules["keypoints"], list):
        raise ValueError("'keypoints' should be a list.")

    # Validate the 'angle_rules' list
    if not isinstance(rules["angle_rules"], list):
        raise ValueError("'angle_rules' should be a list.")

    # Validate each angle rule
    for angle_rule in rules["angle_rules"]:
        if not all(k in angle_rule for k in ["joints", "min_angle", "max_angle", "feedback", "sides"]):
            raise ValueError("Each angle rule must contain 'joints', 'min_angle', 'max_angle', 'feedback', and 'sides'.")

        if not isinstance(angle_rule["joints"], list):
            raise ValueError("'joints' in each angle rule should be a list.")
        if not isinstance(angle_rule["sides"], dict):
            raise ValueError("'sides' in each angle rule should be a dictionary.")

        # Validate 'sides' structure
        for side in ["left", "right"]:
            if side in angle_rule["sides"]:
                if not isinstance(angle_rule["sides"][side], list):
                    raise ValueError(f"'{side}' in 'sides' should be a list.")
                if len(angle_rule["sides"][side]) != len(angle_rule["joints"]):
                    raise ValueError(f"The number of joints in '{side}' should match the number of joints.")

        # Validate angles
        if not isinstance(angle_rule["min_angle"], (int, float)) or not isinstance(angle_rule["max_angle"], (int, float)):
            raise ValueError("Both 'min_angle' and 'max_angle' should be numbers.")
        if angle_rule["min_angle"] >= angle_rule["max_angle"]:
            raise ValueError("'min_angle' should be smaller than 'max_angle'.")

        # Validate feedback
        if not isinstance(angle_rule["feedback"], str):
            raise ValueError("'feedback' should be a string.")

    print("Exercise rules validated successfully.")

def load_training_data(exercise_type, base_dir="data"):
    """
    Load labeled keypoints data from the .npy files.

    Args:
        base_dir (str): Path to the folder containing `correct.npy` and `incorrect.npy`.
        exercise_type (str): Type of exercise (e.g., 'pushup', 'squat').

    Returns:
        dict: A dictionary containing 'correct' and 'incorrect' PoseSequence lists.
    """
    correct_folder = os.path.join(base_dir, exercise_type, "landmarks/correct")
    incorrect_folder = os.path.join(base_dir, exercise_type, "landmarks/incorrect")
    
    if not os.path.exists(correct_folder) or not os.path.exists(incorrect_folder):
        print(f"Training data not found for exercise '{exercise_type}'.")
        return

    correct_sequences = []
    for file in os.listdir(correct_folder):
        if file.endswith(".npy"):
            correct_path = os.path.join(correct_folder, file)
            correct_sequences.append(load_posesequence(correct_path))

    incorrect_sequences = []
    for file in os.listdir(incorrect_folder):
        if file.endswith(".npy"):
            incorrect_path = os.path.join(incorrect_folder, file)
            incorrect_sequences.append(load_posesequence(incorrect_path))

    dict = {"correct": correct_sequences, "incorrect": incorrect_sequences}
    return dict
