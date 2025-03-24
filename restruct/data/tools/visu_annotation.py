import argparse
import csv
import json
import cv2
import numpy as np
import os
from collections import defaultdict

def load_annotations(csv_file):
    """Charge toutes les annotations depuis le fichier CSV."""
    print(f"Chargement des annotations depuis {csv_file}")
    annotations_dict = defaultdict(list)
    
    if not os.path.exists(csv_file):
        print(f"ERREUR: Le fichier {csv_file} n'existe pas")
        return annotations_dict
        
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            print("En-têtes du CSV:", reader.fieldnames)
            row_count = 0
            for row in reader:
                row_count += 1
                filename = row['filename']
                region_shape = json.loads(row['region_shape_attributes'])
                region_attrs = json.loads(row['region_attributes'])
                
                if region_shape['name'] == 'polygon':
                    points = np.array(list(zip(
                        region_shape['all_points_x'],
                        region_shape['all_points_y']
                    )), np.int32)
                    annotations_dict[filename].append({
                        'points': points,
                        'hold_type': region_attrs.get('hold_type', 'unknown')
                    })
        print(f"Nombre de lignes traitées: {row_count}")
        print(f"Nombre d'images annotées: {len(annotations_dict)}")
    except Exception as e:
        print(f"ERREUR lors de la lecture du fichier CSV: {str(e)}")
    
    return annotations_dict

def visualize_annotations(image_path, annotations):
    """Affiche l'image avec les annotations."""
    print(f"Chargement de l'image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Impossible de charger l'image: {image_path}")

    # Dessine chaque annotation
    for ann in annotations:
        points = ann['points']
        # Dessine le polygone en vert
        cv2.polylines(img, [points], True, (0, 255, 0), 2)

    return img

def main():
    parser = argparse.ArgumentParser(description='Visualisation des annotations de prises d\'escalade')
    parser.add_argument('image_dir', help='Chemin vers le répertoire contenant les images')
    parser.add_argument('annotations_file', help='Chemin vers le fichier d\'annotations (CSV)')
    
    args = parser.parse_args()
    
    print(f"Répertoire des images: {args.image_dir}")
    print(f"Fichier d'annotations: {args.annotations_file}")
    
    if not os.path.exists(args.image_dir):
        print(f"ERREUR: Le répertoire {args.image_dir} n'existe pas")
        return
    
    # Charge toutes les annotations
    annotations_dict = load_annotations(args.annotations_file)
    
    if not annotations_dict:
        print("Aucune annotation trouvée dans le fichier")
        return
    
    # Liste des images annotées
    annotated_images = list(annotations_dict.keys())
    current_index = 0
    
    if not annotated_images:
        print("Aucune image annotée trouvée")
        return
    
    print("\nNavigation:")
    print("- 's' : image suivante")
    print("- 'p' : image précédente")
    print("- 'q' : quitter")
    
    while True:
        current_image = annotated_images[current_index]
        image_path = os.path.join(args.image_dir, current_image)
        
        if not os.path.exists(image_path):
            print(f"Image non trouvée: {image_path}")
            current_index = (current_index + 1) % len(annotated_images)
            continue
        
        try:
            # Affiche l'image avec les annotations
            img = visualize_annotations(image_path, annotations_dict[current_image])
            cv2.imshow(f'Image {current_index + 1}/{len(annotated_images)}: {current_image}', img)
            
            # Attend une touche
            key = cv2.waitKey(0)
            
            if key == ord('q'):  # Quitter
                break
            elif key == ord('s'):  # Image suivante
                current_index = (current_index + 1) % len(annotated_images)
            elif key == ord('p'):  # Image précédente
                current_index = (current_index - 1) % len(annotated_images)
        except Exception as e:
            print(f"ERREUR lors de l'affichage de l'image {current_image}: {str(e)}")
            current_index = (current_index + 1) % len(annotated_images)
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
