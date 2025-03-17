import os
import csv
import json
from PIL import Image

def has_valid_annotations(annotation):
    """
    Vérifie si l'annotation contient des coordonnées de polygone valides.
    
    Args:
        annotation (list): Une ligne d'annotation du fichier CSV
        
    Returns:
        bool: True si l'annotation contient des coordonnées valides, False sinon
    """
    try:
        # Vérifie si region_shape_attributes existe et contient des points
        if annotation[5]:  # region_shape_attributes
            region_attrs = json.loads(annotation[5])
            if region_attrs and "all_points_x" in region_attrs:
                # Vérifie si les listes de points ne sont pas vides
                return len(region_attrs["all_points_x"]) > 0 and len(region_attrs["all_points_y"]) > 0
    except (json.JSONDecodeError, IndexError):
        pass
    return False

def convert_dataset(input_dir, output_dir, annotation_file, output_annotation_file, prefix, target_size=(512, 683)):
    """
    Convertit un ensemble de données d'images et leurs annotations.
    
    Args:
        input_dir (str): Répertoire contenant les images d'origine
        output_dir (str): Répertoire de sortie pour les images redimensionnées
        annotation_file (str): Chemin vers le fichier d'annotations CSV d'origine
        output_annotation_file (str): Chemin vers le fichier d'annotations CSV de sortie
        prefix (str): Préfixe à ajouter aux noms de fichiers
        target_size (tuple): Dimensions cibles pour le redimensionnement (largeur, hauteur)
    """
    # Crée le répertoire de sortie s'il n'existe pas
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Charge les annotations existantes si le fichier existe
    existing_annotations = {}
    if os.path.exists(output_annotation_file):
        with open(output_annotation_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_annotations[row['filename']] = row

    # Ouvre le fichier d'annotations d'origine
    annotations = []
    header = None
    with open(annotation_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Sauvegarde l'en-tête
        for row in reader:
            annotations.append(row)

    # Prépare le fichier de sortie
    with open(output_annotation_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)  # Écrit l'en-tête
        
        # Pour chaque annotation
        for annotation in annotations:
            # Vérifie si l'annotation contient des coordonnées valides
            if not has_valid_annotations(annotation):
                print(f"L'image {annotation[0]} n'a pas d'annotations valides, on la saute")
                continue
                
            input_filename = annotation[0]
            input_path = os.path.join(input_dir, input_filename)
            
            # Vérifie si l'image existe
            if not os.path.exists(input_path):
                print(f"L'image {input_path} n'existe pas, on la saute")
                continue
                
            # Génère le nouveau nom de fichier
            new_filename = f"{prefix}-{os.path.splitext(input_filename)[0]}.png"
            output_path = os.path.join(output_dir, new_filename)
            
            # Si l'image a déjà été traitée, on passe
            if new_filename in existing_annotations:
                print(f"L'image {new_filename} existe déjà dans les annotations")
                continue
            
            try:
                # Ouvre et redimensionne l'image
                with Image.open(input_path) as img:
                    # Obtient les dimensions originales
                    original_width, original_height = img.size
                    
                    # Redimensionne l'image
                    resized_img = img.resize(target_size, Image.LANCZOS)
                    resized_img.save(output_path, 'PNG')
                    
                    # Calcule les facteurs d'échelle
                    width_scale = target_size[0] / original_width
                    height_scale = target_size[1] / original_height
                    
                    # Met à jour les annotations
                    new_row = annotation.copy()
                    new_row[0] = new_filename  # Nouveau nom de fichier
                    
                    # Met à jour les coordonnées des régions si elles existent
                    if annotation[5]:  # region_shape_attributes
                        region_attrs = json.loads(annotation[5])
                        if region_attrs and "all_points_x" in region_attrs:
                            # Met à l'échelle les points x et y
                            region_attrs["all_points_x"] = [int(x * width_scale) for x in region_attrs["all_points_x"]]
                            region_attrs["all_points_y"] = [int(y * height_scale) for y in region_attrs["all_points_y"]]
                            new_row[5] = json.dumps(region_attrs)
                    
                    writer.writerow(new_row)
                    print(f"Image {input_filename} convertie en {new_filename}")
                    
            except Exception as e:
                print(f"Erreur lors du traitement de {input_filename}: {str(e)}")

if __name__ == "__main__":
    # Demande les paramètres à l'utilisateur
    input_dir = input("Entrez le chemin du répertoire contenant les images : ")
    output_dir = input("Entrez le chemin du répertoire de sortie pour les images : ")
    annotation_file = input("Entrez le chemin du fichier d'annotations CSV : ")
    output_annotation_file = input("Entrez le chemin du fichier d'annotations de sortie : ")
    prefix = input("Entrez le préfixe à ajouter aux noms de fichiers : ")
    
    # Appelle la fonction de conversion
    convert_dataset(input_dir, output_dir, annotation_file, output_annotation_file, prefix)
