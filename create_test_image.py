import cv2
import numpy as np

# Créer une image de test (400x400 pixels)
image = np.ones((400, 400, 3), dtype=np.uint8) * 255

# Dessiner quelques formes pour simuler des prises d'escalade
cv2.circle(image, (100, 100), 30, (0, 0, 255), -1)  # Rouge
cv2.circle(image, (200, 200), 40, (0, 255, 0), -1)  # Vert
cv2.circle(image, (300, 300), 35, (255, 0, 0), -1)  # Bleu

# Sauvegarder l'image
cv2.imwrite('data/bh/0457.jpg', image) 