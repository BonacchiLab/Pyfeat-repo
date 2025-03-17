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
#folder_path = os.path.abspath("Input_Data")
#p = Path(folder_path)
#directories = [x for x in p.iterdir() if x.is_dir()]
#q = p
#with q.open() as f: f.readline()
#p = PurePath('Input_Data')
#os.fspath(p)
#p = Path('Input_Data')
#folder_path = [x for x in p.iterdir() if x.is_dir()]
#with p.open() as f:
#    f.readline()

p = Path(r'C:\Users\Asus\Documents\recursos\estágio\FACS_repositório\Input_data').glob('*/')
files = [x for x in p if x.is_file()]
#detect single face 
single_face_prediction=detector.detect_image(files, data_type="image") 



#Exibir resultados 
for imagine in (folder_path):
    print(single_face_prediction.head()) 
    print(single_face_prediction.aus) #actionunits
    print(single_face_prediction.emotions) #emotions
    print(single_face_prediction.poses) #head pose
    print(single_face_prediction.identities) 

figs = single_face_prediction.plot_detections(poses=True)
print(figs) #plot single image

#Save output





