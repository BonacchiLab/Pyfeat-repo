# Importações necessárias
import os
import matplotlib.pyplot as plt
from feat import Detector, Fex
from feat.utils.io import get_test_data_path
from feat.plotting import imshow
from IPython.core.display import Video
import cv2  # Biblioteca para manipulação de vídeo

# Inicialização do detector
detector = Detector()
fex = Fex()

# Caminho para o vídeo 
test_video_path = r"C:\Users\Asus\Downloads\video_teste.mp4"

# Exibir o vídeo
display(Video(test_video_path, embed=True))

# Detectar faces no vídeo, skip_frames melhora desempenho
video_prediction = detector.detect_video(
    test_video_path, data_type="video", skip_frames=20, face_detection_threshold=0.95
)

# Exibir as primeiras linhas da predição
print(video_prediction.head())
print(video_prediction.shape)
print(video_prediction.identities)

# Plotar as emoções ao longo do vídeo
plt.figure(figsize=(15, 10))
axes = video_prediction.emotions.plot(title="Emoções ao longo do vídeo")
plt.show()
plt.close("all")

#Visualizing detection results
figs = video_prediction.plot_detections(poses=True)
print(figs)

#Visualizar landmarks usando o modelo de AU padronizado do Py-Feat
figs2 = video_prediction.plot_detections(faces="aus", muscles=True)
print(figs2)

# Abrir o vídeo com OpenCV
cap = cv2.VideoCapture(test_video_path)

# Verificar se o vídeo foi aberto corretamente
if not cap.isOpened():
    raise ValueError("Não foi possível abrir o vídeo.")

# Redefinir o índice do DataFrame para evitar ambiguidade
video_prediction = video_prediction.reset_index()

# Loop imprimir frames com AUs > 0.8 para cada emoção
for emotion in video_prediction.emotions.columns:
    # Filtrar os frames onde o valor da emoção é maior que 0.8
    filtered_frames = video_prediction[video_prediction.emotions[emotion] > 0.8]
    
    if not filtered_frames.empty:
        print(f"Frames com {emotion} > 0.8:")
        print(filtered_frames[['frame', emotion]])  # Exibir os frames e o valor da emoção
        
        # Exibir os frames filtrados
        max_frames_to_process = 5  # Limitar o número de frames processados
        for i, frame_number in enumerate(filtered_frames['frame']):
            if i >= max_frames_to_process:
                print("Limite de frames processados atingido.")
                break
            # Verificar se o frame está dentro do alcance do vídeo
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_number >= total_frames:
                print(f"Frame {frame_number} está fora do alcance do vídeo.")
                continue
            # Configurar o vídeo para o frame específico
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                print(f"Não foi possível capturar o frame {frame_number}.")
                continue
            # Obter a identidade associada ao frame
            identity = video_prediction.identities[frame_number] if frame_number in video_prediction.index else "Desconhecido"
            
            # Converter o frame de BGR para RGB (para exibição com matplotlib)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Exibir o frame como imagem
            plt.figure(figsize=(20, 16))
            plt.imshow(frame_rgb)
            plt.title(f"Frame {frame_number} - {emotion} > 0.8 - Identidade: {identity}")
            plt.axis('off')
            plt.pause(0.001)  # Exibir sem bloquear o loop
            plt.close('all')  # Fechar a figura para liberar memória
    else:
        print(f"Nenhum frame com {emotion} > 0.8.")

# Libertar o vídeo
cap.release()