import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="Scrap Detector", layout="wide")

def analyze_scrap_detection(model_path, image_pil):
    # Convert PIL image to OpenCV format (RGB -> BGR)
    original_img_rgb = np.array(image_pil)
    original_img = cv2.cvtColor(original_img_rgb, cv2.COLOR_RGB2BGR)
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        cv2.imwrite(tmp.name, original_img)
        image_path = tmp.name

    # Load model
    model = YOLO(model_path)
    results = model(image_path)

    annotated_img = original_img_rgb.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    scrap_data = []

    for result in results:
        if result.masks is not None:
            for i, mask in enumerate(result.masks.xy):
                confidence = result.boxes.conf[i].item()
                class_name = result.names[int(result.boxes.cls[i])]
                polygon = np.array(mask, dtype=np.int32)
                area = cv2.contourArea(polygon)
                perimeter = cv2.arcLength(polygon, True)
                x, y, width, height = cv2.boundingRect(polygon)

                scrap_info = {
                    'id': i + 1,
                    'class': class_name,
                    'confidence': confidence,
                    'polygon': polygon,
                    'area': area,
                    'perimeter': perimeter,
                    'width': width,
                    'height': height,
                    'bbox': (x, y, width, height)
                }
                scrap_data.append(scrap_info)

                color = colors[i % len(colors)]
                cv2.fillPoly(annotated_img, [polygon], color)
                cv2.polylines(annotated_img, [polygon], True, color, 2)
                cv2.rectangle(annotated_img, (x, y), (x + width, y + height), color, 2)
                text = f"ID:{i+1} {class_name} {confidence:.2f}"
                cv2.putText(annotated_img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return scrap_data, annotated_img

# Streamlit UI
st.title(" Scrap Detection App with YOLOv8")
st.markdown("Analyse d’image segmentée avec mesures (aire, périmètre...)")

model_path = st.text_input("Chemin vers le modèle YOLOv8 (.pt)", value="runs/segment/train/weights/last.pt")
uploaded_img = st.file_uploader("🖼️ Téléverse une image", type=["jpg", "jpeg", "png"])

if uploaded_img and os.path.exists(model_path):
    image_pil = Image.open(uploaded_img).convert("RGB")
    st.image(image_pil, caption="Image originale", use_column_width=True)

    with st.spinner("🔍 Analyse en cours..."):
        scrap_data, annotated_img = analyze_scrap_detection(model_path, image_pil)

    st.image(annotated_img, caption="Image annotée", use_column_width=True)

    st.markdown("---")
    st.subheader(" Résultats")

    total_area = sum([s['area'] for s in scrap_data])
    avg_conf = np.mean([s['confidence'] for s in scrap_data]) if scrap_data else 0

    st.write(f"Total détecté : {len(scrap_data)} scraps")
    st.write(f" Surface totale : {total_area:.2f} pixels²")
    st.write(f" Confiance moyenne : {avg_conf:.3f}")

    for s in scrap_data:
        with st.expander(f"🔍 Scrap {s['id']} - {s['class']} (conf: {s['confidence']:.2f})"):
            st.write(f"-  Aire : {s['area']:.2f} px²")
            st.write(f"-  Périmètre : {s['perimeter']:.2f} px")
            st.write(f"-  Dimensions : {s['width']}x{s['height']} px")
            st.write(f"-  BBox : x={s['bbox'][0]}, y={s['bbox'][1]}")

    # Téléchargement image annotée
    annotated_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
    _, img_encoded = cv2.imencode(".jpg", annotated_bgr)
    st.download_button("⬇ Télécharger l’image annotée", img_encoded.tobytes(), file_name="annotated.jpg", mime="image/jpeg")

else:
    st.info(" Charge une image et indique le chemin vers ton modèle YOLOv8.")

