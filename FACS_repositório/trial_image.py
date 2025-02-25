from feat import Detector
import matplotlib.pyplot as plt

detector = Detector()

## Processing a single image ##
#Load the image
from feat.utils.io import get_test_data_path
from feat.plotting import imshow
import os

# Helper to point to the test data folder #
test_data_dir = get_test_data_path()

# Get the full path
single_face_img_path = os.path.join(test_data_dir, "single_face.jpg")

# Plot it
imshow(single_face_img_path)

#Import Fex Dataclass
from feat import Fex

fex = Fex() # <-- Special dataframe that stores and operates on detector output

#First analysis #
single_face_prediction = detector.detect_image(single_face_img_path, data_type="image") and detector.detect_image(input())
single_face_prediction

# Show results #
print(single_face_prediction)
print(single_face_prediction.head())
print(single_face_prediction.aus) 
print(single_face_prediction.emotions)
print(single_face_prediction.poses)
print(single_face_prediction.identities)


figs = single_face_prediction.plot_detections(poses=True)

print(figs)

single_face_prediction.to_csv("output.csv", index=False)

from feat.utils.io import read_feat

input_prediction = read_feat("output.csv")

print(input_prediction)