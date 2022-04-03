from Emmanuel.IMG.detector import face_recognition
from eggdriver import put, green, white, yellow

put("""
Bienvenida/o a la aplicacion de reconocimiento facial.
A continuación ingrese la carpeta donde se encuentran las fotos que desea comparar:
""" + white, green)
folder = input()
put("""
Deberá ingresar el nombre de la persona que desea comparar
Opciones disponibles:
yasna
valentina

A continuación ingrese el nombre de la persona que desea comparar:
""" + white, green)
person = input()

face_recognition(enrolled_path="Emmanuel/IMG/" + person, searching_path=folder, th=0.55)
