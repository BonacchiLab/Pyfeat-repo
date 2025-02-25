#Importações necessárias
from feat import Detector
import matplotlib.pyplot as plt
from feat.utils.io import get_test_data_path
from feat.plotting import imshow #show image
import os
from feat import Fex
from IPython.core.display import Video #show video

#Variáveis necessárias
detector = Detector()
fex = Fex()

test_data_dir = get_test_data_path()
test_video_path = os.path.join(test_data_dir, "WolfgangLanger_Pexels.mp4")

# Show video

Video(test_video_path, embed=False)

video_prediction = detector.detect_video(
    test_video_path, data_type="video", skip_frames=24, face_detection_threshold=0.95
)
video_prediction.head()