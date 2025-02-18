#Importações necessárias
from feat import Detector 
import matplotlib.pyplot as plt #plot image if needed
from feat.utils.io import get_test_data_path #path
from feat.plotting import imshow #show image
import os #module 
from feat import Fex #dataclass
from IPython.core.display import Video #show video
from feat.utils.io import read_feat #read csv file

#Variáveis necessárias
detector = Detector()
fex = Fex()

# Get the full path
single_face_img_path = os.path.join("Input_Data")

imshow(single_face_img_path)
