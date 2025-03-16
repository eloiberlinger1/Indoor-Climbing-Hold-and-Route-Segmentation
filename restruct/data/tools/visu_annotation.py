import argparse
import csv
import json
import cv2
import numpy as np

def load_annotations(csv_file, target_image):
    """Charge les annotations pour une image spécifique depuis le fichier CSV."""
    annotations = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['filename'] == target_image:
                region_shape = json.loads(row['region_shape_attributes'])
                region_attrs = json.loads(row['region_attributes'])
                
                if region_shape['name'] == 'polygon':
                    points = np.array(list(zip(
                        region_shape['all_points_x'],
                        region_shape['all_points_y']
                    )), np.int32)
                    annotations.append({
                        'points': points,
                        'hold_type': region_attrs.get('hold_type', 'unknown')
                    })
    return annotations

def visualize_annotations(image_path, annotations):
    """Affiche l'image avec les annotations."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Impossible de charger l'image: {image_path}")

    # Dessine chaque annotation
    for ann in annotations:
        points = ann['points']
        # Dessine le polygone en vert
        cv2.polylines(img, [points], True, (0, 255, 0), 2)

    # Affiche l'image
    cv2.imshow('Image avec annotations', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Visualisation des annotations de prises d\'escalade')
    parser.add_argument('image_path', help='Chemin vers l\'image')
    parser.add_argument('csv_path', help='Chemin vers le fichier CSV d\'annotations')
    
    args = parser.parse_args()
    
    # Extrait le nom du fichier de l'image depuis le chemin complet
    image_filename = args.image_path.split('/')[-1]
    
    # Charge les annotations
    annotations = load_annotations(args.csv_path, image_filename)
    
    if not annotations:
        print(f"Aucune annotation trouvée pour l'image {image_filename}")
        return
    
    # Visualise les annotations
    visualize_annotations(args.image_path, annotations)

if __name__ == '__main__':
    main()
