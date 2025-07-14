import cv2
import numpy as np
import random

def augmenter_image(image):
    # 1. Flip horizontal avec 50% de chance
    if random.random() < 0.5:
        image = cv2.flip(image, 1)

    # 2. Rotation aléatoire entre -20 et 20 degrés
    if random.random() < 0.5:
        angle = random.uniform(-20, 20)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        image = cv2.warpAffine(image, M, (w, h))

    # 3. Zoom (crop + resize)
    if random.random() < 0.3:
        zoom_factor = random.uniform(0.8, 1.0)
        h, w = image.shape[:2]
        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
        y1 = random.randint(0, h - new_h)
        x1 = random.randint(0, w - new_w)
        image = image[y1:y1+new_h, x1:x1+new_w]
        image = cv2.resize(image, (w, h))

    # 4. Variation de la luminosité
    if random.random() < 0.5:
        factor = random.uniform(0.6, 1.4)
        image = np.clip(image * factor, 0, 255).astype(np.uint8)

    # 5. Ajout de bruit gaussien
    if random.random() < 0.2:
        noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
        image = cv2.add(image, noise)

    return image
