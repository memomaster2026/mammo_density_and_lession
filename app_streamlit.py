"""
================================================================================
APPLICATION STREAMLIT - TEST DU MODELE DE CLASSIFICATION DE DENSITE MAMMAIRE
================================================================================
Cette application permet de :
  - Charger une ou plusieurs images (mammographies)
  - Les redimensionner AUTOMATIQUEMENT à la taille exigée par le modèle
    (peu importe la taille/format d'origine de l'image fournie)
  - Appliquer le même prétraitement que lors de l'entraînement (CLAHE)
  - Faire classer les images par le modèle
  - Afficher les probabilités par classe, la classe prédite, la confiance,
    la marge top1-top2, et un statut "accepté / incertain" selon les seuils

Lancement :
    streamlit run app_streamlit.py
================================================================================
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow import keras
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURATION (identique au script d'entraînement / d'évaluation)
# ==============================================================================

CLASSES = ["ACR_A", "ACR_B", "ACR_C", "ACR_D"]
NUM_CLASSES = len(CLASSES)

IMG_SIZE = 512          # <-- Taille exigée par le modèle (entrée du réseau)
CHANNELS = 3

USE_CLAHE = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

CONFIDENCE_THRESHOLD = 0.65
MARGIN_THRESHOLD = 0.15

# Chemin par défaut vers le modèle (modifiable dans la barre latérale)
DEFAULT_MODEL_PATH = "modele_densite_final.keras"


# ==============================================================================
# PRETRAITEMENT (identique à l'entraînement)
# ==============================================================================

def appliquer_clahe(image_np: np.ndarray) -> np.ndarray:
    """Applique un CLAHE sur le canal de luminance (espace LAB)."""
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def pretraiter_image(pil_image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """
    Prend une image PIL (n'importe quelle taille/format fournie par l'utilisateur),
    la convertit en RGB, la REDIMENSIONNE automatiquement à IMG_SIZE x IMG_SIZE
    (taille exigée par le modèle), puis applique le même prétraitement CLAHE
    qu'à l'entraînement.

    Retourne :
      - img_pour_modele : tableau float32 prêt à être envoyé au modèle
      - img_pour_affichage : tableau uint8 (0-255) pour l'aperçu dans l'UI
    """
    # 1. Forcer en RGB (gère les images en niveaux de gris, RGBA, CMYK, etc.)
    img_rgb = np.array(pil_image.convert("RGB"))

    # 2. Redimensionnement automatique à la taille exigée par le modèle
    img_resized = cv2.resize(
        img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR
    )

    # 3. Prétraitement CLAHE (identique à l'entraînement)
    if USE_CLAHE:
        img_final = appliquer_clahe(img_resized)
    else:
        img_final = img_resized

    img_pour_modele = img_final.astype(np.float32)
    img_pour_affichage = img_final.astype(np.uint8)

    return img_pour_modele, img_pour_affichage


# ==============================================================================
# CHARGEMENT DU MODELE (mis en cache pour ne pas le recharger à chaque clic)
# ==============================================================================

@st.cache_resource(show_spinner="Chargement du modèle...")
def charger_modele(chemin_modele: str):
    model = keras.models.load_model(chemin_modele, compile=False)
    return model


def predire(model, batch_images: np.ndarray) -> np.ndarray:
    return model.predict(batch_images, verbose=0)


# ==============================================================================
# INTERFACE STREAMLIT
# ==============================================================================

st.set_page_config(
    page_title="Test - Classification densité mammaire (ACR)",
    layout="wide",
)

st.title("🩻 Test du modèle de classification de densité mammaire (ACR A-D)")
st.caption(
    "Les images fournies sont automatiquement redimensionnées à "
    f"{IMG_SIZE}x{IMG_SIZE} px (taille exigée par le modèle), quelle que "
    "soit leur taille ou leur format d'origine."
)

# ---- Barre latérale : configuration ----------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    chemin_modele = st.text_input("Chemin du fichier modèle (.keras)", value=DEFAULT_MODEL_PATH)

    st.markdown("---")
    st.subheader("Prétraitement")
    use_clahe_ui = st.checkbox("Appliquer CLAHE (recommandé, identique à l'entraînement)", value=USE_CLAHE)
    USE_CLAHE = use_clahe_ui

    st.markdown("---")
    st.subheader("Seuils de décision")
    CONFIDENCE_THRESHOLD = st.slider("Seuil de confiance minimum", 0.0, 1.0, CONFIDENCE_THRESHOLD, 0.01)
    MARGIN_THRESHOLD = st.slider("Marge minimum (top1 - top2)", 0.0, 1.0, MARGIN_THRESHOLD, 0.01)

    st.markdown("---")
    st.caption(f"Taille d'entrée du modèle : **{IMG_SIZE} x {IMG_SIZE} px** (redimensionnement automatique)")

# ---- Chargement du modèle ---------------------------------------------------
if not os.path.exists(chemin_modele):
    st.warning(
        f"⚠️ Le fichier modèle `{chemin_modele}` est introuvable. "
        "Indique le bon chemin dans la barre latérale."
    )
    st.stop()

try:
    model = charger_modele(chemin_modele)
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {e}")
    st.stop()

st.success("✅ Modèle chargé avec succès.")

# ---- Upload des images -------------------------------------------------------
fichiers = st.file_uploader(
    "Dépose une ou plusieurs images de mammographie",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
    accept_multiple_files=True,
)

if not fichiers:
    st.info("👆 Charge une ou plusieurs images pour lancer la classification.")
    st.stop()

st.markdown(f"### {len(fichiers)} image(s) chargée(s)")

# ---- Prétraitement + redimensionnement automatique de toutes les images -----
images_modele = []
images_affichage = []
noms_fichiers = []
tailles_originales = []

for f in fichiers:
    pil_img = Image.open(f)
    tailles_originales.append(pil_img.size)  # (largeur, hauteur) d'origine
    img_modele, img_affichage = pretraiter_image(pil_img)
    images_modele.append(img_modele)
    images_affichage.append(img_affichage)
    noms_fichiers.append(f.name)

batch = np.stack(images_modele)

# ---- Prédiction ---------------------------------------------------------------
with st.spinner("Classification en cours..."):
    probs = predire(model, batch)

# ---- Construction des résultats -----------------------------------------------
resultats = []
for nom, taille_orig, p in zip(noms_fichiers, tailles_originales, probs):
    ordre = np.argsort(p)[::-1]
    idx_top1, idx_top2 = ordre[0], ordre[1]
    confiance = float(p[idx_top1])
    marge = float(p[idx_top1] - p[idx_top2])
    accepte = confiance >= CONFIDENCE_THRESHOLD and marge >= MARGIN_THRESHOLD
    resultats.append({
        "fichier": nom,
        "taille_originale": f"{taille_orig[0]}x{taille_orig[1]}",
        "taille_redimensionnee": f"{IMG_SIZE}x{IMG_SIZE}",
        "classe_predite": CLASSES[idx_top1] if accepte else "incertain",
        "classe_top1_sans_rejet": CLASSES[idx_top1],
        "confiance": round(confiance, 4),
        "marge_top1_top2": round(marge, 4),
        **{f"proba_{c}": round(float(p[i]), 4) for i, c in enumerate(CLASSES)},
    })

df = pd.DataFrame(resultats)

# ---- Affichage détaillé par image ---------------------------------------------
st.markdown("### 🔍 Résultats détaillés par image")

for i, row in df.iterrows():
    col_img, col_info = st.columns([1, 2])
    with col_img:
        st.image(
            images_affichage[i],
            caption=f"{row['fichier']} — original {row['taille_originale']} → {row['taille_redimensionnee']}",
            use_container_width=True,
        )
    with col_info:
        statut = "✅ Accepté" if row["classe_predite"] != "incertain" else "⚠️ Incertain"
        st.markdown(f"**Classe prédite (sans rejet) :** `{row['classe_top1_sans_rejet']}`")
        st.markdown(f"**Décision finale ({statut}) :** `{row['classe_predite']}`")
        st.markdown(f"**Confiance :** {row['confiance']*100:.2f}%  |  **Marge top1-top2 :** {row['marge_top1_top2']*100:.2f} pts")

        fig, ax = plt.subplots(figsize=(5, 2))
        ax.bar(CLASSES, [row[f"proba_{c}"] for c in CLASSES], color="steelblue")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probabilité")
        for j, c in enumerate(CLASSES):
            ax.text(j, row[f"proba_{c}"], f"{row[f'proba_{c}']:.2f}", ha="center", va="bottom", fontsize=8)
        st.pyplot(fig)
        plt.close(fig)
    st.markdown("---")

# ---- Tableau récapitulatif + distribution --------------------------------------
st.markdown("### 📊 Récapitulatif")
st.dataframe(df, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Répartition (classe la plus probable, sans rejet)**")
    st.bar_chart(df["classe_top1_sans_rejet"].value_counts().reindex(CLASSES).fillna(0))
with col2:
    st.markdown("**Répartition avec rejet (incertain inclus)**")
    st.bar_chart(df["classe_predite"].value_counts().reindex(CLASSES + ["incertain"]).fillna(0))

# ---- Export CSV -------------------------------------------------------------
csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Télécharger les résultats (CSV)",
    data=csv,
    file_name="predictions_streamlit.csv",
    mime="text/csv",
)
