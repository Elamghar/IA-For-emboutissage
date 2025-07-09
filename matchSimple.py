import json

# Charger la base des pièces
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
    # Surface OK ?
    if piece["surface_requise"] > scrap["surface"]:
        return False

    # Dimensions OK ? (dans les deux orientations)
    p_L = piece["dimensions"]["longueur"]
    p_l = piece["dimensions"]["largeur"]
    s_L = scrap["dimensions"]["longueur"]
    s_l = scrap["dimensions"]["largeur"]

    return (
        (p_L <= s_L and p_l <= s_l) or
        (p_L <= s_l and p_l <= s_L)
    )

# Matching
for scrap in scraps_detectes:
    print(f"\n Scrap ID {scrap['id']} – Surface: {scrap['surface']} px² – Dimensions: {scrap['dimensions']['longueur']}×{scrap['dimensions']['largeur']} px")
    print(" Pièces faisables :")

    found = False
    for piece in base_pieces:
        if piece_peut_etre_fabriquee(piece, scrap):
            print(f"  {piece['nom']} – Surface requise: {piece['surface_requise']} px² – {piece['dimensions']['longueur']}×{piece['dimensions']['largeur']} px")
            found = True

    if not found:
        print(" Aucune pièce compatible.")
