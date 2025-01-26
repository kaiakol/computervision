import os
from parse import load_ps

def load_training_data(training_folder):
    """
    Load training data for exercises from a folder structure.

    Args:
        training_folder (str): Path to the folder containing training data.

    Returns:
        dict: A dictionary with exercise types as keys and training sequences as values.
    """
    training_data = {}
    exercises = os.listdir(training_folder)
    for exercise in exercises:
        exercise_path = os.path.join(training_folder, exercise)
        if os.path.isdir(exercise_path):
            training_data[exercise] = {
                "correct": [],
                "incorrect": []
            }
            for label in ["correct", "incorrect"]:
                label_path = os.path.join(exercise_path, label)
                if os.path.exists(label_path):
                    for file in os.listdir(label_path):
                        if file.endswith(".npy"):
                            sequence = load_ps(os.path.join(label_path, file))
                            training_data[exercise][label].append(sequence)
    return training_data
