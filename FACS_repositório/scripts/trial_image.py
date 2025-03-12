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
single_face_img_path = os.path.join(test_data_dir, "0-f1-su-ph.jpg")

# Plot it
imshow(single_face_img_path)

#Import Fex Dataclass
from feat import Fex

fex = Fex() # <-- Special dataframe that stores and operates on detector output

#First analysis #
single_face_prediction = detector.detect_image(single_face_img_path, data_type="image")
single_face_prediction

# Show results #
print(single_face_prediction)
print(single_face_prediction.head())
print(single_face_prediction.aus) 
print(single_face_prediction.emotions)
print(single_face_prediction.poses)
print(single_face_prediction.identities)

#Visualizing detection results
#figs = single_face_prediction.plot_detections(poses=True)

#print(figs)

#single_face_prediction.to_csv("output.csv", index=False)

from feat.utils.io import read_feat
#retirar dados
#input_prediction = read_feat("output.csv")

#single_face_prediction.iplot_detections(bounding_boxes=True, emotions=True)

#Visualize a face using Py-Feat’s standardized AU landmark model
figs1 = single_face_prediction.plot_detections(faces="aus", muscles=True)
print(figs1)

#Generate an interective ploty figure that lets you interectavly enable/disable detector outputs
#(FEX OBJECT HAS NO ATTRIVUTE IPLOT_DETECTIONS)single_face_prediction.iplot_detections(bounding_boxes=True, emotions=True)

##MULTIPLE FACES##
#multi_face_image_path = os.path.join(test_data_dir, "multi_face.jpg")
#multi_face_prediction = detector.detect_image(multi_face_image_path, data_type="image")

#Show results
#print(multi_face_prediction)

#Visualizing detection results
#figs2 = multi_face_prediction.plot_detections(add_titles=False)

##MULTIPLE IMAGES##
#img_list = [single_face_img_path, multi_face_image_path]

#mixed_prediction = detector.detect_image(img_list, batch_size=1, data_type="image")
#mixed_prediction

#Visualizing detection results
#figs3 = mixed_prediction.plot_detections()
#print(figs3)