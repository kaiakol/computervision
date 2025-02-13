import numpy as np
import mediapipe as mp

class PoseSequence:
    def __init__(self, sequence):
        self.poses = [Pose(parts) for parts in sequence]
        self.perspective = self.detect_perspective()
        self.min_angles = {}
        self.max_angles = {}
        self.motion_ranges = {}
        self.compute_motion_ranges()

    def compute_motion_ranges(self):
        """
        Compute and normalize the total motion range for each joint over the sequence.
        The motion range is calculated as the cumulative Euclidean distance traveled by each joint.
        Normalized values range from 0 to 1.

        Updates:
            self.motion_ranges[joint_name]: The normalized total distance each joint traveled in the sequence.
        """
        joint_positions = {joint: [] for joint in Pose.PART_NAMES}
        raw_motion_ranges = {}

        # Collect joint positions throughout the sequence
        for pose in self.poses:
            for joint in Pose.PART_NAMES:
                part = getattr(pose, joint)
                if part.exists:
                    joint_positions[joint].append((part.x, part.y))

        # Compute total motion range for each joint
        for joint, positions in joint_positions.items():
            if len(positions) > 1:  # Need at least two positions to compute motion
                total_distance = sum(
                    np.sqrt((positions[i+1][0] - positions[i][0])**2 + (positions[i+1][1] - positions[i][1])**2)
                    for i in range(len(positions) - 1)
                )
                raw_motion_ranges[joint] = total_distance
            else:
                raw_motion_ranges[joint] = 0  # No movement detected or insufficient data

        # Normalize the motion ranges to be between 0 and 1
        max_motion_range = max(raw_motion_ranges.values(), default=1)  # Avoid division by zero

        self.motion_ranges = {
            joint: (raw_motion_ranges[joint] / max_motion_range) if max_motion_range > 0 else 0
            for joint in Pose.PART_NAMES
        }

    def detect_perspective(self):
        """
        Detect the recording perspective: left, right, or front.
        """
        left_visibility, right_visibility = 0, 0
        left_x_sum, right_x_sum = 0, 0
        left_count, right_count = 0, 0

        for pose in self.poses:
            for attr, part in pose:
                if part.exists:
                    if attr.startswith("l"):  # Left side keypoints
                        left_visibility += part.c
                        left_x_sum += part.x
                        left_count += 1
                    elif attr.startswith("r"):  # Right side keypoints
                        right_visibility += part.c
                        right_x_sum += part.x
                        right_count += 1

        # Compute average visibility and x-coordinate symmetry
        left_avg_visibility = left_visibility / left_count if left_count > 0 else 0
        right_avg_visibility = right_visibility / right_count if right_count > 0 else 0
        symmetry = abs(left_avg_visibility - right_avg_visibility)

        # Determine perspective
        if symmetry < 0.2:  # Symmetry threshold to classify as "front"
            return "front"
        elif left_avg_visibility > right_avg_visibility:
            return "left"
        else:
            return "right"
        
    def compute_joint_angle_statistics(self, joint_names):
        """
        Compute the minimum and maximum joint angles for a sequence of poses.
        """
        angles = []
        for pose in self.poses:
            angle = pose.compute_joint_angle(joint_names)
            if angle is not None:
                angles.append(angle)
        
        if angles:
            self.min_angles[tuple(joint_names)] = np.min(angles)
            self.max_angles[tuple(joint_names)] = np.max(angles)
        else:
            self.min_angles[tuple(joint_names)] = None
            self.max_angles[tuple(joint_names)] = None

    def normalize_perspective(self):
        """
        Normalize the pose sequence to the right-side view.
        """
        if self.perspective == "left":
            for pose in self.poses:
                for attr, part in pose:
                    if attr.startswith("l"):
                        setattr(pose, attr, Part([-part.x, -part.y, part.c]))

class Pose:
    PART_NAMES = [
        "nose", "lshoulder", "rshoulder", "lelbow", "relbow", "lwrist", "rwrist",
        "lhip", "rhip", "lknee", "rknee", "lankle", "rankle", "lheel", "rheel",
        "lfootindex", "rfootindex"
    ]
    
    def __init__(self, parts):
        for name, vals in zip(Pose.PART_NAMES, parts):
            if len(vals) != 3:
                raise ValueError(f"Expected 3 values for {name}, got {vals}")
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
    
    def compute_joint_angle(self, joint_names):
        """
        Compute the angle between three joints in the current pose.

        Args:
            joint_names (list): A list of three joint names defining the angle
                                (e.g., ["lshoulder", "lelbow", "lwrist"]).

        Returns:
            float: The computed angle in degrees, or None if any joint is missing.
        """
        if len(joint_names) != 3:
            raise ValueError("Exactly 3 joint names must be provided.")

        joint1, joint2, joint3 = joint_names
        part1 = getattr(self, joint1)
        part2 = getattr(self, joint2)
        part3 = getattr(self, joint3)

        if not (part1.exists and part2.exists and part3.exists):
            return None

        # Compute and normalize vectors
        vec1 = np.array([part1.x - part2.x, part1.y - part2.y])
        vec2 = np.array([part3.x - part2.x, part3.y - part2.y])
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)

        # Compute angle between vectors
        dot_product = np.dot(vec1_norm, vec2_norm)
        angle = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
        return angle
    
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
