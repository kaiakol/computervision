import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

mp_pose = mp.solutions.pose

class PoseTrainer:
    def __init__(self):
        self.pose = mp_pose.Pose()

    def estimate_pose(self, video_path):
        """Step 4.3: Extract keypoints using Mediapipe Pose."""
        cap = cv2.VideoCapture(video_path)
        all_frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)

            if results.pose_landmarks:
                keypoints = [
                    {
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                        "visibility": lm.visibility,
                    }
                    for lm in results.pose_landmarks.landmark
                ]
                all_frames.append(keypoints)

        cap.release()
        return all_frames

    def normalize_keypoints(self, frames):
        """Step 4.4: Normalize keypoints based on torso length."""
        normalized_frames = []

        for frame in frames:
            neck = np.array([frame[mp_pose.PoseLandmark.NOSE.value]["x"],
                             frame[mp_pose.PoseLandmark.NOSE.value]["y"]])
            left_hip = np.array([frame[mp_pose.PoseLandmark.LEFT_HIP.value]["x"],
                                 frame[mp_pose.PoseLandmark.LEFT_HIP.value]["y"]])
            right_hip = np.array([frame[mp_pose.PoseLandmark.RIGHT_HIP.value]["x"],
                                  frame[mp_pose.PoseLandmark.RIGHT_HIP.value]["y"]])

            torso_length = np.linalg.norm(neck - left_hip) + np.linalg.norm(neck - right_hip) / 2

            normalized_frame = {
                k: {
                    "x": (v["x"] - neck[0]) / torso_length,
                    "y": (v["y"] - neck[1]) / torso_length,
                    "visibility": v["visibility"],
                }
                for k, v in enumerate(frame)
            }
            normalized_frames.append(normalized_frame)

        return normalized_frames

    def detect_perspective(self, frames):
        """Step 4.5: Determine which side is performing the exercise."""
        left_visibility = sum(frame[mp_pose.PoseLandmark.LEFT_ELBOW.value]["visibility"] for frame in frames)
        right_visibility = sum(frame[mp_pose.PoseLandmark.RIGHT_ELBOW.value]["visibility"] for frame in frames)
        return "left" if left_visibility > right_visibility else "right"

    def evaluate_bicep_curl(self, frames):
        """Step 4.6: Evaluate bicep curl geometry."""
        for frame in frames:
            upper_arm = np.array([frame[mp_pose.PoseLandmark.SHOULDER.value]["x"], 
                                  frame[mp_pose.PoseLandmark.SHOULDER.value]["y"]]) - \
                        np.array([frame[mp_pose.PoseLandmark.ELBOW.value]["x"], 
                                  frame[mp_pose.PoseLandmark.ELBOW.value]["y"]])
            forearm = np.array([frame[mp_pose.PoseLandmark.WRIST.value]["x"], 
                                frame[mp_pose.PoseLandmark.WRIST.value]["y"]]) - \
                      np.array([frame[mp_pose.PoseLandmark.ELBOW.value]["x"], 
                                frame[mp_pose.PoseLandmark.ELBOW.value]["y"]])

            angle = np.degrees(np.arccos(np.dot(upper_arm, forearm) / 
                                         (np.linalg.norm(upper_arm) * np.linalg.norm(forearm))))
            if angle < 70:
                print("Good form")
            else:
                print("Incomplete curl")

    def dtw_distance(self, seq1, seq2):
        """Step 4.7: Compute DTW distance between two keypoint sequences."""
        distance, _ = fastdtw(seq1, seq2, dist=euclidean)
        return distance

    def classify_form(self, input_sequence, training_sequences):
        """Classify form using DTW and nearest neighbor."""
        distances = [(label, self.dtw_distance(input_sequence, seq)) for label, seq in training_sequences]
        return min(distances, key=lambda x: x[1])[0]


# Example usage:
trainer = PoseTrainer()

# Step 4.2: Load and process video
video_path = "exercise_video.mp4"
raw_keypoints = trainer.estimate_pose(video_path)

# Step 4.4: Normalize keypoints
normalized_keypoints = trainer.normalize_keypoints(raw_keypoints)

# Step 4.5: Detect perspective
perspective = trainer.detect_perspective(normalized_keypoints)
print("Dominant side performing exercise:", perspective)

# Step 4.6: Evaluate bicep curl
trainer.evaluate_bicep_curl(normalized_keypoints)

# Step 4.7: Machine learning evaluation
# Example training data: [("correct", sequence1), ("incorrect", sequence2)]
training_data = [("correct", normalized_keypoints), ("incorrect", normalized_keypoints)]
result = trainer.classify_form(normalized_keypoints, training_data)
print("Form classification:", result)
