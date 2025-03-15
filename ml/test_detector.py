import argparse
import matplotlib.pyplot as plt
from model import HoldDetector
import os

def main():
    # Configuration des arguments
    parser = argparse.ArgumentParser(description='Test du détecteur de prises d\'escalade')
    parser.add_argument('--image', type=str, required=True, help='Chemin vers l\'image à analyser')
    parser.add_argument('--output', type=str, default='output', help='Préfixe pour les fichiers de sortie')
    parser.add_argument('--device', type=str, default='cpu', help='Device à utiliser (cpu ou cuda)')
    args = parser.parse_args()
    
    # Chemin du fichier de poids
    weights_path = "ml/weights/yolov8_hold_detector.pt"
    
    # Vérification des fichiers
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Fichier de poids non trouvé: {weights_path}")
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image non trouvée: {args.image}")
    
    # Initialisation du détecteur
    print("Initialisation du détecteur...")
    detector = HoldDetector(weights_path, args.device)
    
    # Prédiction
    print(f"Analyse de l'image: {args.image}")
    results = detector.predict(args.image)
    
    # Sauvegarde des prédictions en JSON
    json_path = f"{args.output}_predictions.json"
    detector.save_predictions_to_json(results["outputs"], json_path)
    print(f"Prédictions sauvegardées dans: {json_path}")
    
    # Affichage des résultats
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Image originale
    ax1.imshow(results["original_image"][:, :, ::-1])
    ax1.axis('off')
    ax1.set_title('Image originale')
    
    # Image avec détections
    ax2.imshow(results["annotated_image"][:, :, ::-1])
    ax2.axis('off')
    ax2.set_title('Détections')
    
    # Sauvegarde de la figure
    plt.savefig(f"{args.output}_visualization.png")
    print(f"Visualisation sauvegardée dans: {args.output}_visualization.png")
    
    # Affichage si demandé
    plt.show()

if __name__ == "__main__":
    main() 