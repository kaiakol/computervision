## Pose Trainer with MediaPipe

The purpose of this application is to be able to use a keypoint detection model such as MediaPipe for exercise form correction and exercise detection.

The model uses both a heuristic evaluation method and a machine learning model to give feedback on whether an exercise was performed with correct form or not, and give feedback on how the user can improve their form. The model also detects what exercise is being done.

Supported exercises are squat, push up, pull up, jumping jack and russian twist.

### Install required packages:

`pip install mediapipe opencv-python numpy`

### Running the application:

`python main.py --video [video_path] --exercise [exercise_type] 2>/dev/null`

Example:
`python main.py --video data/videos/pushup_correct_001.mp4 --exercise pushup 2>/dev/null`
