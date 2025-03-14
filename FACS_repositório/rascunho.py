#Importações necessárias
from feat import Detector 
import matplotlib.pyplot as plt #plot image if needed
from feat.utils.io import get_test_data_path #path
from feat.plotting import imshow #show image
import os #module 
from feat import Fex #dataclass
from IPython.core.display import Video #show video
from feat.utils.io import read_feat #read csv file
from pathlib import Path #get file path

#Variáveis necessárias
detector = Detector()
fex = Fex()

# Get the full path
folder_path = os.path.abspath("Input_Data")

#for dirpath, _, filenames in os.walk(folder_path):
#    for file in filenames:
#        if file.lower().endswith(('.jpg', '.png', '.jpeg')):  # add videos !!!
#            file_path = os.path.join(dirpath, file)
#           print(f"Processing: {file_path}")           # Detectar face na imagem
#            single_face_prediction = detector.detect_image(file_path, data_type="image")
#            
#            # Exibir imagem
#            imshow(file_path)

p = Path(folder_path)
[x for x in p.iterdir() if x.is_dir()]



#Exibir resultados 
print(single_face_prediction.head()) 
print(single_face_prediction.aus) #actionunits
print(single_face_prediction.emotions) #emotions
print(single_face_prediction.poses) #head pose
print(single_face_prediction.identities) 

figs = single_face_prediction.plot_detections(poses=True)
print(figs) #plot single image




