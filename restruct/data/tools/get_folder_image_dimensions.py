import os
from PIL import Image

def get_image_dimensions(directory):
    # Vérifie si le répertoire existe
    if not os.path.isdir(directory):
        print(f"Le répertoire {directory} n'existe pas.")
        return

    # Liste tous les fichiers dans le répertoire
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Vérifie si le fichier est une image
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                print(f"Image: {filename}, Dimensions: {width}x{height}")
        except IOError:
            # Si le fichier n'est pas une image, passe au suivant
            print(f"{filename} n'est pas une image valide.")

if __name__ == "__main__":
    # Demande à l'utilisateur de spécifier le répertoire
    directory = input("Veuillez entrer le chemin du répertoire contenant les images : ")
    get_image_dimensions(directory)
