import cv2
import mediapipe as mp
import numpy as np

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points.
    a, b, c are points in (x, y) format.
    """
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)  # Get angle in radians
    return np.degrees(angle)  # Convert to degrees

# Initialize MediaPipe Pose and drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Exercise configurations
exercises = {
    "pushup": {
        "landmarks": [...],
        "down_angle": ...,
        "up_angle": ...
    },
    "squat": {
        "landmarks": [mp_pose.PoseLandmark.LEFT_HIP, 
                      mp_pose.PoseLandmark.LEFT_KNEE, 
                      mp_pose.PoseLandmark.LEFT_ANKLE,
                      mp_pose.PoseLandmark.RIGHT_HIP,
                      mp_pose.PoseLandmark.RIGHT_KNEE,
                      mp_pose.PoseLandmark.RIGHT_ANKLE],
        "down_angle": 70,
        "up_angle": 160
    },
    "situp": {
        "landmarks": [...],
        "down_angle": ...,
        "up_angle": ...
    },
    "lunge": {
        "landmarks": [...],
        "down_angle": ...,
        "up_angle": ...
    },
    "russian_twist": {
        "landmarks": [...],
        "down_angle": ...,
        "up_angle": ...
    },
    "back_extension": {
        "landmarks": [...],
        "down_angle": ...,
        "up_angle": ...
    },
    "calf_raise": {
        "landmarks": [...],
        "down_angle": ...,
        "up_angle": ...
    },
    "dips": {
        "landmarks": [...],
        "down_angle": ...,
        "up_angle": ...
    },
    "jumping_jacks": {
        "landmarks": [...],
        "down_angle": ...,
        "up_angle": ...
    },
    "burpees": {
        "landmarks": [...],
        "down_angle": ...,
        "up_angle": ...
    },
}

# Variables for tracking
current_exercise = "squat"  # Default exercise
counter = 0
stage = None  # "down" or "up"

# Start video capture
video_path = "../data/videos/pushup_correct_001.mp4"
cap = cv2.VideoCapture(video_path)

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Video has ended or cannot be read")
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect pose landmarks
        results = pose.process(frame_rgb)

        # Extract landmarks and calculate angles
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get the current exercise configuration
            exercise = exercises[current_exercise]
            l1, l2, l3 = exercise["landmarks"]

            # Get key landmarks for the current exercise
            point1 = landmarks[l1]
            point2 = landmarks[l2]
            point3 = landmarks[l3]

            # Calculate the angle
            angle = calculate_angle(point1, point2, point3)

            # Display the angle on the frame
            cv2.putText(frame, f"{current_exercise.capitalize()} Angle: {int(angle)}",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Exercise detection logic
            if angle < exercise["down_angle"]:
                stage = "down"
            if angle > exercise["up_angle"] and stage == "down":
                stage = "up"
                counter += 1  # Increment counter

            # Display exercise count and stage
            cv2.putText(frame, f"{current_exercise.capitalize()} Count: {counter}",
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Stage: {stage}",
                        (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Draw pose landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

        # Show the processed frame
        cv2.imshow('Exercise Tracker', frame)

        # Quit when 'q' is pressed
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
