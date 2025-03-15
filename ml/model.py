from ultralytics import YOLO
import cv2
import numpy as np
import json

class HoldDetector:
    def __init__(self, weights_path, device='cpu'):
        """
        Initialise le détecteur de prises d'escalade avec YOLOv8.
        
        Args:
            weights_path (str): Chemin vers les poids du modèle
            device (str): Device à utiliser ('cpu' ou 'cuda')
        """
        self.model = YOLO(weights_path)
        self.device = device
        self.classes = ["hold", "volume"]
        
    def predict(self, image_path):
        """
        Effectue la prédiction sur une image.
        
        Args:
            image_path (str): Chemin vers l'image à analyser
            
        Returns:
            dict: Dictionnaire contenant les prédictions et l'image annotée
        """
        # Lecture de l'image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Impossible de lire l'image: {image_path}")
            
        # Prédiction
        results = self.model(img, device=self.device)[0]
        
        # Création de l'image annotée
        img_holds = img.copy()
        for box in results.boxes:
            # Récupération des coordonnées
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            # Dessin de la boîte
            color = (0, 255, 0) if cls == 0 else (255, 0, 0)  # Vert pour hold, Rouge pour volume
            cv2.rectangle(img_holds, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Ajout du label
            label = f"{self.classes[cls]} {conf:.2f}"
            cv2.putText(img_holds, label, (int(x1), int(y1)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return {
            "outputs": results,
            "original_image": img,
            "annotated_image": img_holds
        }
        
    def save_predictions_to_json(self, outputs, output_path):
        """
        Sauvegarde les prédictions au format JSON.
        
        Args:
            outputs: Sortie du modèle YOLO
            output_path (str): Chemin où sauvegarder le fichier JSON
        """
        predictions = []
        
        for box in outputs.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            pred = {
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "score": float(box.conf[0]),
                "category_id": int(box.cls[0]),
                "category_name": self.classes[int(box.cls[0])]
            }
            predictions.append(pred)
            
        with open(output_path, 'w') as f:
            json.dump({"predictions": predictions}, f, indent=2) 