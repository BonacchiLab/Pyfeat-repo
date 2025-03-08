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
folder_path = os.path.abspath("Input_Data")

for dirpath, _, filenames in os.walk(folder_path):
    for file in filenames:
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):  # add videos !!!
            file_path = os.path.join(dirpath, file)
            print(f"Processing: {file_path}")
            
            # Detectar face na imagem
            prediction = detector.detect_image(file_path, data_type="image")
            
            # Exibir imagem
            imshow(file_path)


