import cv2
import mediapipe as mp
import numpy as np

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    """
    Calculate the angle between three points.
    a, b, c are points in (x, y) format.
    """
    a = np.array([a.x, a.y])  # Point A
    b = np.array([b.x, b.y])  # Point B (joint)
    c = np.array([c.x, c.y])  # Point C

    # Calculate vectors
    ba = a - b
    bc = c - b

    # Calculate the cosine of the angle
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)  # Get angle in radians
    return np.degrees(angle)  # Convert to degrees

# Initialize MediaPipe Pose and drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Squat detection thresholds
DOWN_ANGLE = 70  # Angle below this is considered a squat down
UP_ANGLE = 160   # Angle above this is considered standing up

# Variables for squat counting
counter = 0
stage = None  # "down" or "up"

# Start video capture
cap = cv2.VideoCapture(0)

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
            print("Ignoring empty camera frame.")
            continue

        # Flip the frame horizontally for a mirror-like view
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect pose landmarks
        results = pose.process(frame_rgb)

        # Extract landmarks and calculate angles
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get key landmarks for squat detection
            hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
            ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]

            # Calculate knee angle
            angle = calculate_angle(hip, knee, ankle)

            # Display the angle on the frame
            cv2.putText(frame, f"Knee Angle: {int(angle)}",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Squat detection logic
            if angle < DOWN_ANGLE:
                stage = "down"
            if angle > UP_ANGLE and stage == "down":
                stage = "up"
                counter += 1  # Increment squat counter

            # Display squat count and stage
            cv2.putText(frame, f"Squats: {counter}",
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
        cv2.imshow('Squat Detection', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
