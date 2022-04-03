from detector import face_recognition
from eggdriver import put, green, white

put("""
Bienvenida/o a la aplicacion de reconocimiento facial.
A continuación ingrese la dirección de la carpeta donde se encuentran las fotos que desea comparar:
""" + white, green)
folder = input()

people_dir = ""
if people_dir == "":
    put("""
Ingrese la dirección de la carpeta en que se encuentran las carpetas de personas
""" + white, green)
    people_dir = input()

available_people = []

if available_people == []:
    availability = "no people available"
else:
    availability = "/n".join(available_people)

put(f"""
Deberá ingresar el nombre de la persona que desea comparar
Opciones disponibles:
{availability}

A continuación ingrese el nombre de la persona que desea comparar:
""" + white, green)
person = input()

face_recognition(enrolled_path=people_dir + "/" + person, searching_path=folder, th=0.55)
