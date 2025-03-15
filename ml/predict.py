import cv2
import matplotlib.pyplot as plt
from model import HoldDetector
import os

def visualize_predictions(image, predictions, output_path=None):
    """
    Visualise les prédictions sur l'image.
    
    Args:
        image: Image originale (format BGR)
        predictions: Sortie du modèle YOLO
        output_path: Chemin pour sauvegarder la visualisation (optionnel)
    """
    # Création de l'image annotée
    img_holds = image.copy()
    
    # Dessin des boîtes et labels
    for box in predictions.boxes:
        # Récupération des coordonnées
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        
        # Couleur selon la classe (vert pour hold, rouge pour volume)
        color = (0, 255, 0) if cls == 0 else (255, 0, 0)
        
        # Dessin de la boîte
        cv2.rectangle(img_holds, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Ajout du label
        label = f"{'hold' if cls == 0 else 'volume'} {conf:.2f}"
        cv2.putText(img_holds, label, (int(x1), int(y1)-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Création de la figure avec matplotlib
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Image originale
    ax1.imshow(image[:, :, ::-1])
    ax1.axis('off')
    ax1.set_title('Image originale')
    
    # Image avec détections
    ax2.imshow(img_holds[:, :, ::-1])
    ax2.axis('off')
    ax2.set_title('Détections')
    
    # Sauvegarde si un chemin est fourni
    if output_path:
        plt.savefig(output_path)
        print(f"Visualisation sauvegardée dans: {output_path}")
    
    # Affichage
    plt.show()
    
    return img_holds

def main():
    # Chemin du fichier de poids
    weights_path = "ml/weights/yolov8_hold_detector.pt"
    
    # Vérification du fichier de poids
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Fichier de poids non trouvé: {weights_path}")
    
    # Initialisation du détecteur
    print("Initialisation du détecteur...")
    detector = HoldDetector(weights_path)
    
    # Chemin de l'image de test
    image_path = "data/sm/45escalade_block_image_1.png"
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image non trouvée: {image_path}")
    
    # Lecture de l'image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Impossible de lire l'image: {image_path}")
    
    # Prédiction
    print(f"Analyse de l'image: {image_path}")
    results = detector.predict(image_path)
    
    # Sauvegarde des prédictions en JSON
    json_path = "output_predictions.json"
    detector.save_predictions_to_json(results["outputs"], json_path)
    print(f"Prédictions sauvegardées dans: {json_path}")
    
    # Visualisation
    visualize_predictions(image, results["outputs"], "output_visualization.png")

if __name__ == "__main__":
    main() 