from ultralytics import YOLO
import os
from pathlib import Path

def main():
    # Obtenir le chemin absolu du répertoire de travail
    workspace_dir = Path.cwd()
    
    # Créer un nouveau modèle YOLOv8
    model = YOLO('yolov8n.pt')  # Charger le modèle pré-entraîné YOLOv8n
    
    # Configuration de l'entraînement
    data_yaml = f"""
    path: {workspace_dir}/data  # Chemin vers le dossier de données
    train: images/train  # Images d'entraînement
    val: images/val  # Images de validation
    
    # Classes
    names:
      0: hold
      1: volume
    """
    
    # Sauvegarder la configuration
    yaml_path = workspace_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(data_yaml)
    
    # Créer le dossier weights s'il n'existe pas
    weights_dir = workspace_dir / 'ml' / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Entraîner le modèle
    results = model.train(
        data=str(yaml_path),
        epochs=100,
        imgsz=640,
        batch=16,
        device='mps',  # ou 'cuda' si disponible
        project=str(weights_dir),
        name='hold_detector'
    )
    
    # Sauvegarder le modèle final
    model.export(format='pt', filename=str(weights_dir / 'yolov8_hold_detector.pt'))

if __name__ == "__main__":
    main() 