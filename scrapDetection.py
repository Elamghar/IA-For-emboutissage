import cv2
import numpy as np
import matplotlib.pyplot as plt

# Charger l'image
image = cv2.imread("OP40.png")  # Remplace par ton image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)

# TROUVER TOUS LES CONTOURS + HIERARCHIE
contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

image_output = image.copy()
scrap_data = []

# Analyse des contours
for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h)

    # Vérifier si c'est un "enfant" d'un contour (donc un trou)
    if hierarchy[0][i][3] != -1:  # Si le parent existe, c’est une découpe interne

        # Filtrer par taille et forme
        if area > 3000 and 0.3 < aspect_ratio < 3.5:
            scrap_data.append({
                "index": i,
                "surface": round(area, 2),
                "perimetre": round(perimeter, 2),
                "bounding_box": (x, y, w, h)
            })

            cv2.drawContours(image_output, [contour], -1, (0, 255, 0), 2)
            cv2.putText(image_output, f"#{i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

# Affichage
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(image_output, cv2.COLOR_BGR2RGB))
plt.title("Découpes internes détectées (scraps)")
plt.axis("off")
plt.show()

# Détails
for scrap in scrap_data:
    print(f"Scrap #{scrap['index']} – Surface: {scrap['surface']} px² – BBox: {scrap['bounding_box']}")
