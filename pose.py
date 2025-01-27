import cv2
import numpy as np
import mediapipe as mp

class PoseSequence:
    def __init__(self, sequence):
        self.poses = []
        for parts in sequence:
            self.poses.append(Pose(parts))

        # Normalize poses based on the average torso length
        torso_lengths = []
        for pose in self.poses:
            midpoint_shoulder = (pose.lshoulder + pose.rshoulder) / 2
            midpoint_hip = (pose.lhip + pose.rhip) / 2
            mean_torso = Part.dist(midpoint_shoulder, midpoint_hip)
            torso_lengths.append(mean_torso)
        
        mean_torso = np.mean(torso_lengths) if torso_lengths else 1.0

        for pose in self.poses:
            for attr, part in pose:
                setattr(pose, attr, part / mean_torso)

class Pose:
    PART_NAMES = [
        "nose", "lshoulder", "rshoulder", "lelbow", "relbow", "lwrist", "rwrist",
        "lhip", "rhip", "lknee", "rknee", "lankle", "rankle", "lheel", "rheel",
        "lfootindex", "rfootindex"
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

    def __add__(self, other):
        if not (self.exists and other.exists):
            return Part([0, 0, 0])  # Return a "nonexistent" Part if either doesn't exist
        return Part([self.x + other.x, self.y + other.y, min(self.c, other.c)])  # Combine confidences conservatively

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
            mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            mp_pose.PoseLandmark.LEFT_ELBOW.value,
            mp_pose.PoseLandmark.RIGHT_ELBOW.value,
            mp_pose.PoseLandmark.LEFT_WRIST.value,
            mp_pose.PoseLandmark.RIGHT_WRIST.value,
            mp_pose.PoseLandmark.LEFT_HIP.value,
            mp_pose.PoseLandmark.RIGHT_HIP.value,
            mp_pose.PoseLandmark.LEFT_KNEE.value,
            mp_pose.PoseLandmark.RIGHT_KNEE.value,
            mp_pose.PoseLandmark.LEFT_ANKLE.value,
            mp_pose.PoseLandmark.RIGHT_ANKLE.value,
            mp_pose.PoseLandmark.LEFT_HEEL.value,
            mp_pose.PoseLandmark.RIGHT_HEEL.value,
            mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value,
            mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value
        ]
    ]
    return keypoints