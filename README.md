## Pose Trainer with MediaPipe

The purpose of this application is to be able to use a keypoint detection model such as MediaPipe for exercise form correction and repetition counting. 

The model uses both a heuristic evaluation method and a machine learning model to give feedback on whether an exercise was performed with correct form or not, and give feedback on how the user can improve their form.

### Install required packages:
`pip install mediapipe opencv-python numpy`

### Running the application:
`python main.py --video [video_path] --exercise [exercise_type]`

Example:
`python main.py --video data/videos/pushup_correct_001.mp4 --exercise pushup`
