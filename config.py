import sys
from pathlib import Path
from hand_tracker import HandTracker
from hand_tracker2 import HandTracker as HandTracker2

sys.path.append(Path("..").resolve().as_posix())

from utils.comp_vision import bokeh_viz

palm_model_path = "./models/palm_detection_without_custom_op.tflite"
landmark_model_path = "./models/hand_landmark.tflite"
anchors_path = "./data/anchors.csv" 

tracker = HandTracker(palm_model_path, landmark_model_path, anchors_path)
tracker2 = HandTracker2(palm_model_path, landmark_model_path, anchors_path)
