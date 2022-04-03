# arcface: Librería de Reconocimiento Facial
# mtcnn: Librería de Detección de Caras
# opencv-contrib-python: Librería de Computer Vision

import eggdriver
#eggdriver.init() # Actualiza pip e instala todo + astropy necesario para usar Arcface preentrenada

import cv2                            # Librería OpenCV
import matplotlib.pyplot as plt       # Librería para Visualización (gráficas, imágenes, etc.)
import numpy as np                    # Librería de Matemáticas
from   tqdm.auto import tqdm          # Librería para barra de progreso
import os, fnmatch                    # Librería para leer directorios
from   arcface import ArcFace         # Librería de Reconocimiento Facial
from   mtcnn.mtcnn import MTCNN       # Librería de Detección de Caras 

detector = MTCNN()                    # Carga del Detector de Caras
arcface  = ArcFace.ArcFace()          # Carga del Reconocedor Facial

def display(img, title, *args):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    if len(args) > 0:
        for a in args:
            plt.plot(a[0], a[1], c=a[2])
    plt.show()

def boundary_box(y_lower_bound, y_upper_bound, x_lower_bound, x_upper_bound):
    y = [y_lower_bound, y_lower_bound, y_upper_bound, y_upper_bound, y_lower_bound]
    x = [x_lower_bound, x_upper_bound, x_upper_bound, x_lower_bound, x_lower_bound]
    return y, x

def boundary2verteces(boundary_box):
    x1, y1, w, h = boundary_box[0], boundary_box[1], boundary_box[2], boundary_box[3]
    x2, y2 = x1 + w, y1 + h
    return (y1, y2, x1, x2)

def obtain_boxes(img, boundary_color = "red", return_crops = False):
    face_locations = detector.detect_faces(img)
    n = len(face_locations)
    boxes = []
    crops = []
    for i in range(n):
        bb = face_locations[i]['box']
        bb_verteces = boundary2verteces(bb)
        cy, cx = boundary_box(*bb_verteces)
        boxes.append([cx, cy, boundary_color])
        if return_crops:
            y1, y2, x1, x2 = bb_verteces
            crops.append(img[y1:y2,x1:x2,:])
    boxes = tuple(boxes)
    if return_crops:
        return boxes, crops
    return boxes

def compare_imgs(img1, img2, th = 0.6):
    d1 = arcface.calc_emb(img1)
    d2 = arcface.calc_emb(img2)
    score = np.dot(d1,d2.T)
    return score > th

def dirfiles(img_path,img_ext):
    img_names = fnmatch.filter(sorted(os.listdir(img_path)),img_ext)
    return img_names

def detected_in(detect_list, index):
    for detected in detect_list:
        if detected[index]:
            return True
    return False

def detect_faces(fotos):
    m = len(fotos)
    BB = []
    for k in range(m):
        img_path = ipath+fotos[k]
        I = cv2.imread(img_path)                  # <= cargar en I la imagen k de la galeria
        print(str(k)+'/'+str(m)+' detecting faces in image '+ img_path)
        face_locations = detector.detect_faces(I) # <= cargar en face_locations las caras detectadas en I
        n = len(face_locations)                   # <= cargar en n el número de caras en la imagen I
        print(str(n)+' faces detected')
        for i in range(n):
            bb = face_locations[i]['box']           # <= cargar en bbox el bounding box i detectado
            bb_verteces = boundary2verteces(bb)     # CODIGO MIO
            x1   = bb_verteces[2]                   # <= cargar el valor x1 del bounding box
            y1   = bb_verteces[0]                   # <= cargar el valor y1 del bounding box
            x2   = bb_verteces[3]                   # <= cargar el valor x2 del bounding box
            y2   = bb_verteces[1]                   # <= cargar el valor y2 del bounding box
            BB.append([k,x1,y1,x2,y2])
    N = len(BB)
    print(str(N)+' faces detected in '+str(m)+' images')
    return BB, N

yasna_path1 = "photos/yasna_ejemplo.jpg"
yasna_path2 = "photos/yasna_ejemplo2.jpg"
x = cv2.imread(yasna_path1)
boxes, crops  = obtain_boxes(x, "red", True)
x1 = crops[0]
x = cv2.imread(yasna_path2)
boxes, crops  = obtain_boxes(x, "red", True)
x2 = crops[0]
d1 = arcface.calc_emb(x1)
d2 = arcface.calc_emb(x2)
descriptors = [d1, d2]
display(x1, "Imagen Original")

ipath = 'photos/'
fotos = dirfiles(ipath,'*.jpg')
print('Galería de fotos:', fotos)

BB, N = detect_faces(fotos)
Y = np.zeros((N,512))
for j in tqdm(range(N)):
    (k,x1,y1,x2,y2) = BB[j]    # <= Cargar el elemento j de BB
    img_path = ipath+fotos[k]  # <= nombre (con su path) de la foto k de la galería
    I = cv2.imread(img_path)   # <= lectura de la foto 
    J = I[y1:y2,x1:x2,:]       # <= cropping de la cara detectada
    ej = arcface.calc_emb(J)   # <= descriptor ArcFace de la cara detectada
    Y[j,:] = ej

th = 0.55                    # <= umbral de comparación
score1 = np.dot(Y, d1)       # <= comparación de descriptores 1
score2 = np.dot(Y, d2)       # <= comparación de descriptores 2
scores = [score1, score2]
detect1 = [int(s > th) for s in score1]
detect2 = [int(s > th) for s in score2]
detect_list = [detect1, detect2]
print(detect1, detect2)

dim = (256,256) # para hacer resize de la cara detectada
S = cv2.resize(x, dim)
pos = (25,25)
cv2.putText(S,"Enrolled", pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),3)
for k in range(len(fotos)):
    plt.figure(figsize=(15,20))
    img_path = ipath+fotos[k]   # <= nombre (con su path) de la foto k de la galería
    I = cv2.imread(img_path)    # <= lectura de la imagen k de la galeria
    plt.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))
    plt.title(fotos[k])
    for j in range(N):
        (i,x1,y1,x2,y2) = BB[j]   # <= cargar el elemento j de BB
        if i==k and detected_in(detect_list, j):
            for index in range(len(detect_list)):
                detected = detect_list[index]
                score = scores[index]
                if detected[j]:
                    cy, cx = boundary_box(y1, y2, x1, x2) # CODIGO MIO
                    x = cx                  # <= cargar coordenadas x del bounding box
                    y = cy                  # <= cargar coordenadas y del bounding box
                    plt.plot(x,y,c='red')
                    Id = cv2.resize(I[y1:y2,x1:x2,:],dim)
                    cv2.putText(Id,fotos[k], pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),3)
                    S = cv2.hconcat([S,Id])
                    print('face recognized in image '+fotos[k]+' > score = '+str(score[j]))
    plt.show()

plt.figure(figsize=(15,5))
plt.imshow(cv2.cvtColor(S, cv2.COLOR_BGR2RGB))
plt.show()
