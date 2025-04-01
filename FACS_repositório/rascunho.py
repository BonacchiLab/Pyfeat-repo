# Importações necessárias
from feat import Detector
import matplotlib.pyplot as plt  # plot image if needed
from feat.utils.io import get_test_data_path  # path
from feat.plotting import imshow  # show image
import os  # module
from feat import Fex  # dataclass
from IPython.core.display import Video  # show video
from feat.utils.io import read_feat  # read csv file
from pathlib import Path  # get file path
import json
from glob import glob
import seaborn as sns
from tqdm import tqdm

# Inicialização do detector e Fex
detector = Detector()
fex = Fex()


# Diretório de dados de teste
test_data_dir = get_test_data_path()


# Caminho completo para a imagem de teste
single_face_img_path = r"C:\Users\Asus\Downloads\.jpg"

# Caminho do vídeo de teste
test_video_path = r"C:\Users\Asus\Downloads\single_face.mp4"

# Plotar a imagem
imshow(single_face_img_path)

# Detectar características na imagem
single_face_prediction = detector.detect_image(single_face_img_path, data_type="image")

# Exibir resultados

print(single_face_prediction.head())  # Exibir as primeiras linhas
print(single_face_prediction.aus)  # Action Units
print(single_face_prediction.emotions)  # Emoções
print(single_face_prediction.poses)  # Head pose
print(single_face_prediction.identities)  # Identidades

# Plotar detecções com poses
figs = single_face_prediction.plot_detections(poses=True)
plt.show()  # Mostrar o gráfico gerado





