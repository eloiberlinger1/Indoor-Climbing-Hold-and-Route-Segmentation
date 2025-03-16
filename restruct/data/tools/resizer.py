import os
from PIL import Image

def resize_images_in_directory(directory, output_directory, size):
    # Vérifie si le répertoire existe
    if not os.path.isdir(directory):
        print(f"Le répertoire {directory} n'existe pas.")
        return

    # Crée le répertoire de sortie s'il n'existe pas
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Liste tous les fichiers dans le répertoire
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Vérifie si le fichier est une image
        try:
            with Image.open(file_path) as img:
                # Redimensionne l'image
                resized_img = img.resize(size, Image.LANCZOS)
                # Construit le chemin de sortie pour la nouvelle image
                output_path = os.path.join(output_directory, os.path.splitext(filename)[0] + '.png')
                # Sauvegarde la nouvelle image au format PNG
                resized_img.save(output_path, 'PNG')
                print(f"L'image {filename} a été redimensionnée et sauvegardée sous {output_path}")
        except IOError:
            # Si le fichier n'est pas une image, passe au suivant
            print(f"{filename} n'est pas une image valide.")

if __name__ == "__main__":
    # Demande à l'utilisateur de spécifier le répertoire d'entrée et de sortie
    directory = input("Veuillez entrer le chemin du répertoire contenant les images : ")
    output_directory = input("Veuillez entrer le chemin du répertoire de sortie pour les images redimensionnées : ")

    # Appelle la fonction pour redimensionner les images
    resize_images_in_directory(directory, output_directory, (512, 683))
