import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np

# Charger la base de pièces
with open("base_pieces_fabrication.json", "r") as f:
    base_pieces = json.load(f)

# Exemple de scraps détectés
scraps_detectes = [
    {"id": 0, "surface": 10000, "dimensions": {"longueur": 100, "largeur": 100}},
    {"id": 1, "surface": 15000, "dimensions": {"longueur": 160, "largeur": 120}},
    {"id": 2, "surface": 30000, "dimensions": {"longueur": 250, "largeur": 250}},
]

# Fonction de vérification
def piece_peut_etre_fabriquee(piece, scrap):
    if piece["surface_requise"] > scrap["surface"]:
        return False

    p_L, p_l = piece["dimensions"]["longueur"], piece["dimensions"]["largeur"]
    s_L, s_l = scrap["dimensions"]["longueur"], scrap["dimensions"]["largeur"]

    return (
        (p_L <= s_L and p_l <= s_l) or
        (p_L <= s_l and p_l <= s_L)
    )

# Affichage
for scrap in scraps_detectes:
    # Filtrer les pièces compatibles
    compatibles = [p for p in base_pieces if piece_peut_etre_fabriquee(p, scrap)]

    # Nombre total d'images à afficher 
    total = 1 + len(compatibles)

    plt.figure(figsize=(4 * total, 5))

    #  Afficher le scrap comme une boîte grise
    plt.subplot(1, total, 1)
    scrap_img = np.ones((scrap["dimensions"]["largeur"], scrap["dimensions"]["longueur"], 3), dtype=np.uint8) * 230
    plt.imshow(scrap_img)
    plt.title(f"Scrap #{scrap['id']}\n{scrap['dimensions']['longueur']}×{scrap['dimensions']['largeur']} px\nSurface: {scrap['surface']}")
    plt.axis("off")

    # Afficher les pièces compatibles
    for i, piece in enumerate(compatibles):
        img_path = piece["fichier_modele"]
        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            plt.subplot(1, total, i + 2)
            plt.imshow(img)
            plt.title(f"Pièce {piece['nom']}\n{piece['dimensions']['longueur']}×{piece['dimensions']['largeur']} px\nSurf: {piece['surface_requise']}")
            plt.axis("off")
        else:
            print(f"⚠️ Image manquante : {img_path}")

    plt.tight_layout()
    plt.show()
