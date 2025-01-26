import cv2
import numpy as np
import mediapipe as mp

class PoseSequence:
    def __init__(self, sequence):
        self.poses = []
        for parts in sequence:
            self.poses.append(Pose(parts))

        # Normalize poses based on the average torso pixel length
        torso_lengths = []
        for pose in self.poses:
            if pose.neck.exists and pose.lhip.exists:
                torso_lengths.append(Part.dist(pose.neck, pose.lhip))
            if pose.neck.exists and pose.rhip.exists:
                torso_lengths.append(Part.dist(pose.neck, pose.rhip))
        
        mean_torso = np.mean(torso_lengths) if torso_lengths else 1.0

        for pose in self.poses:
            for attr, part in pose:
                setattr(pose, attr, part / mean_torso)

class Pose:
    PART_NAMES = [
        "nose", "neck", "rshoulder", "relbow", "rwrist",
        "lshoulder", "lelbow", "lwrist", "rhip", "rknee", "rankle",
        "lhip", "lknee", "lankle", "reye", "leye", "rear", "lear"
    ]

    def __init__(self, parts):
        for name, vals in zip(Pose.PART_NAMES, parts):
            setattr(self, name, Part(vals))

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

    def __str__(self):
        out = ""
        for name in self.PART_NAMES:
            part = getattr(self, name)
            out += f"{name}: {part.x},{part.y}\n" if part.exists else f"{name}: Not detected\n"
        return out
    
    def print(self, parts):
        out = ""
        for name in parts:
            if name not in self.PART_NAMES:
                raise NameError(f"{name} is not a valid part name.")
            part = getattr(self, name)
            out += f"{name}: {part.x},{part.y}\n" if part.exists else f"{name}: Not detected\n"
        return out

class Part:
    def __init__(self, vals):
        self.x = vals[0]
        self.y = vals[1]
        self.c = vals[2]
        self.exists = self.c > 0.0

    def __floordiv__(self, scalar):
        return self.__truediv__(scalar)

    def __truediv__(self, scalar):
        return Part([self.x / scalar, self.y / scalar, self.c])

    @staticmethod
    def dist(part1, part2):
        if not (part1.exists and part2.exists):
            return 0.0
        return np.sqrt(np.square(part1.x - part2.x) + np.square(part1.y - part2.y))

mp_pose = mp.solutions.pose
pose_estimator = mp_pose.Pose()

def process_mediapipe_results(results):
    """
    Convert Mediapipe landmarks to the format compatible with PoseSequence.
    """
    if not results.pose_landmarks:
        return []

    # Extract keypoints of interest
    landmarks = results.pose_landmarks.landmark
    keypoints = [
        [landmarks[i].x, landmarks[i].y, landmarks[i].visibility]
        for i in [
            mp_pose.PoseLandmark.NOSE.value,
            mp_pose.PoseLandmark.LEFT_SHOULDER.value,  # Neck approximation
            mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            mp_pose.PoseLandmark.RIGHT_ELBOW.value,
            mp_pose.PoseLandmark.RIGHT_WRIST.value,
            mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            mp_pose.PoseLandmark.LEFT_ELBOW.value,
            mp_pose.PoseLandmark.LEFT_WRIST.value,
            mp_pose.PoseLandmark.RIGHT_HIP.value,
            mp_pose.PoseLandmark.RIGHT_KNEE.value,
            mp_pose.PoseLandmark.RIGHT_ANKLE.value,
            mp_pose.PoseLandmark.LEFT_HIP.value,
            mp_pose.PoseLandmark.LEFT_KNEE.value,
            mp_pose.PoseLandmark.LEFT_ANKLE.value,
            mp_pose.PoseLandmark.RIGHT_EYE.value,
            mp_pose.PoseLandmark.LEFT_EYE.value,
            mp_pose.PoseLandmark.RIGHT_EAR.value,
            mp_pose.PoseLandmark.LEFT_EAR.value,
        ]
    ]
    return keypoints


# Example: Process a video frame
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    sequences = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB as required by Mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_estimator.process(frame_rgb)

        # Convert Mediapipe results to PoseSequence format
        keypoints = process_mediapipe_results(results)
        if keypoints:
            sequences.append(keypoints)

    cap.release()
    return PoseSequence(sequences)