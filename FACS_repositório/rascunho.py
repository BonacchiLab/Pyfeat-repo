# Importações necessárias
import os
import json
from feat import Detector, Fex
from feat.utils.io import get_test_data_path, read_feat
from feat.plotting import imshow
from IPython.core.display import Video
import matplotlib.pyplot as plt
from glob import glob

# Inicializar o detector
detector = Detector()

# Caminho relativo para a pasta de entrada
input_data_dir = os.path.join(os.path.dirname(__file__), "Input_data")

# Verificar se a pasta existe
if not os.path.exists(input_data_dir):
    raise FileNotFoundError(f"'input_data' was not {input_data_dir}")

# Carregar configurações do arquivo config.json
config_path = os.path.join(os.path.dirname(__file__), "config.json")
if not os.path.exists(config_path):
    raise FileNotFoundError(f"'config.json' was not found in {config_path}")

with open(config_path, "r") as config_file:
    config = json.load(config_file)

# Determinar se o script deve processar imagens, vídeos ou ambos
process_types = config.get("process_type", ["image", "video"])  # Padrão: ambos

# Obter todos os arquivos na pasta input_data
input_files = glob(os.path.join(input_data_dir, "*"))

# Processar imagens
if "image" in process_types:
    image_files = [f for f in input_files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    for image_path in image_files:
        print(f"Processing images: {image_path}")
        # Exibir a imagem
        imshow(image_path)

        # Detectar características na imagem
        prediction = detector.detect_image(image_path, data_type="image")

        # Exibir resultados
        print(prediction.head())
        print(prediction.aus)  # Action Units
        print(prediction.emotions)  # Emoções
        print(prediction.poses)  # Head pose
        print(prediction.identities)  # Identidades

        # Plotar detecções com poses
        figs = prediction.plot_detections(poses=True)
        plt.show()

# Processar vídeos
if "video" in process_types:
    video_files = [f for f in input_files if f.lower().endswith((".mp4", ".avi", ".mov"))]
    for video_path in video_files:
        print(f"Processing videos: {video_path}")
        # Exibir o vídeo
        display(Video(video_path, embed=True))

        # Detectar características no vídeo
        prediction = detector.detect_video(video_path, data_type="video", skip_frames=20)

        # Exibir resultados
        print(prediction.head())

        # Plotar emoções ao longo do vídeo
        plt.figure(figsize=(15, 10))
        axes = prediction.emotions.plot(title="Emotions throughout the video")
        plt.show()

        # Visualizar detecções no vídeo
        figs = prediction.plot_detections(poses=True)
        plt.show()





