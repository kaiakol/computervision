## Pose Trainer with MediaPipe

The purpose of this application is to be able to use a keypoint detection model such as MediaPipe for exercise form correction and repetition counting. 

The model uses both a heuristic evaluation method and a machine learning model to give feedback on whether an exercise was performed with correct form or not, and give feedback on how the user can improve their form.

Process videos:
python main.py --mode batch_process --input_folder videos --output_folder poses_compressed

Evaluate a video:
python main.py --mode evaluate --video videos/pushup.mp4 --exercise pushup

Evaluate preprocessed keypoints:
python main.py --mode evaluate --video videos/pushup.mp4 --exercise pushup
