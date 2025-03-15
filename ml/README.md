# Détecteur de Prises d'Escalade

Ce module contient le code pour la détection des prises d'escalade dans les images.

## Structure du Projet

```
ml/
├── model.py           # Classe principale du détecteur
├── test_detector.py   # Script de test
├── weights/          # Dossier contenant les poids du modèle
│   └── yolov8_hold_detector.pt
└── README.md
```

## Prérequis

- Python 3.7+
- Ultralytics (YOLOv8)
- OpenCV
- Matplotlib
- PyTorch

## Installation

1. Créer un environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate  # Sur Linux/Mac
# ou
venv\Scripts\activate  # Sur Windows
```

2. Installer les dépendances :
```bash
pip install ultralytics opencv-python matplotlib
```

## Utilisation

Pour tester le détecteur sur une image :

```bash
python test_detector.py --image chemin/vers/image.jpg --output resultat
```

Options :
- `--image` : Chemin vers l'image à analyser (requis)
- `--output` : Préfixe pour les fichiers de sortie (défaut: 'output')
- `--device` : Device à utiliser (cpu ou cuda, défaut: 'cpu')

Le script génère deux fichiers :
1. `{output}_predictions.json` : Contient les détections au format JSON
2. `{output}_visualization.png` : Visualisation des détections

## Format des Prédictions

Le fichier JSON contient un dictionnaire avec une clé "predictions" qui est une liste de détections. Chaque détection contient :
- `bbox` : Coordonnées de la boîte englobante [x1, y1, x2, y2]
- `score` : Score de confiance de la détection
- `category_id` : ID de la catégorie (0: hold, 1: volume)
- `category_name` : Nom de la catégorie

## Visualisation

Les détections sont visualisées avec :
- Boîtes vertes pour les prises (holds)
- Boîtes rouges pour les volumes
- Labels indiquant la catégorie et le score de confiance 