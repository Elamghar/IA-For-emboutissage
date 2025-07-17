import streamlit as st
import json
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ultralytics import YOLO
from io import BytesIO
from PIL import Image

# Charger la base de données des pièces
with open("base_pieces_fabrication.json", "r") as f:
    base_pieces = json.load(f)

# --- Vérifie si une pièce est compatible avec une chute ---
def piece_peut_etre_fabriquee(piece, scrap):
    if piece["surface_requise"] > scrap["surface"]:
        return False
    p_L, p_l = piece["dimensions"]["longueur"], piece["dimensions"]["largeur"]
    s_L, s_l = scrap["dimensions"]["longueur"], scrap["dimensions"]["largeur"]
    return (p_L <= s_L and p_l <= s_l) or (p_L <= s_l and p_l <= s_L)

# --- Analyse l'image avec YOLO ---
def detect_scraps(model, image_np):
    results = model(image_np)
    annotated_img = image_np.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    scraps_detectes = []

    for result in results:
        if result.masks is not None:
            for i, mask in enumerate(result.masks.xy):
                polygon = np.array(mask, dtype=np.int32)
                area = cv2.contourArea(polygon)
                x, y, w, h = cv2.boundingRect(polygon)

                scrap = {
                    "id": i,
                    "surface": area,
                    "dimensions": {"longueur": w, "largeur": h},
                    "polygon": polygon
                }
                scraps_detectes.append(scrap)

                color = colors[i % len(colors)]
                cv2.polylines(annotated_img, [polygon], True, color, 2)
                cv2.putText(annotated_img, f"Scrap #{i} | A={int(area)}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return scraps_detectes, annotated_img

# --- Affiche les résultats dans Streamlit ---
def afficher_resultats_streamlit(scraps_detectes, base_pieces):
    for scrap in scraps_detectes:
        compatibles = [p for p in base_pieces if piece_peut_etre_fabriquee(p, scrap)]
        total = 1 + len(compatibles)

        cols = st.columns(total)
        with cols[0]:
            st.markdown(f"### Scrap #{scrap['id']}")
            st.image(np.ones((scrap["dimensions"]["largeur"], scrap["dimensions"]["longueur"], 3), dtype=np.uint8) * 220,
                     caption=f"{scrap['dimensions']['longueur']}×{scrap['dimensions']['largeur']} px | Surface: {scrap['surface']:.0f}",
                     use_column_width=True)

        for i, piece in enumerate(compatibles):
            with cols[i + 1]:
                if os.path.exists(piece["fichier_modele"]):
                    st.image(piece["fichier_modele"],
                             caption=f"{piece['nom']} | {piece['dimensions']['longueur']}×{piece['dimensions']['largeur']} px | Surf: {piece['surface_requise']}",
                             use_column_width=True)
                else:
                    st.warning(f"Image manquante : {piece['fichier_modele']}")

# --- Application Streamlit ---
st.set_page_config(page_title="Analyse de Chutes", layout="wide")
st.title("Analyse de Chutes avec Détection et Compatibilité")

# Upload image
uploaded_file = st.file_uploader("Choisir une image", type=["png", "jpg", "jpeg"])
model_path = "trainedModel/weights/last.pt"

if uploaded_file:
    st.success(" Image chargée")
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Load YOLO model once
    if "model" not in st.session_state:
        st.session_state.model = YOLO(model_path)

    # Analyse
    with st.spinner("Détection des chutes en cours..."):
        scraps_detectes, annotated_img = detect_scraps(st.session_state.model, image_np)

    # Résultat visuel
    st.subheader(" Image Annotée")
    st.image(annotated_img, channels="RGB")

    # Résultats de compatibilité
    st.subheader(" Pièces Compatibles pour Chaque Chute")
    afficher_resultats_streamlit(scraps_detectes, base_pieces)
