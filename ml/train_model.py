from ultralytics import YOLO
import os

def main():
    # Créer un nouveau modèle YOLOv8
    model = YOLO('yolov8n.pt')  # Charger le modèle pré-entraîné YOLOv8n
    
    # Configuration de l'entraînement
    data_yaml = """
    path: data  # Chemin vers le dossier de données
    train: images/train  # Images d'entraînement
    val: images/val  # Images de validation
    
    # Classes
    names:
      0: hold
      1: volume
    """
    
    # Sauvegarder la configuration
    with open('data.yaml', 'w') as f:
        f.write(data_yaml)
    
    # Entraîner le modèle
    results = model.train(
        data='data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device='cpu',  # ou 'cuda' si disponible
        project='ml/weights',
        name='hold_detector'
    )
    
    # Sauvegarder le modèle final
    model.export(format='pt')

if __name__ == "__main__":
    main() 