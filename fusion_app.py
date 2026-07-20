#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Application d'Analyse Mammographique - Interface Moderne
========================================================
Analyses combinées : Densité mammaire (ACR A/B/C/D) + Lésions (BI-RADS)
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import cv2
import gc
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Union
from ultralytics import YOLO
from pathlib import Path
import time
from io import BytesIO
import base64

# ==========================================================
# CONFIGURATION DE LA PAGE
# ==========================================================

st.set_page_config(
    page_title="MammoAI - Analyse Mammographique",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================================
# CSS MODERNE ET PROFESSIONNEL
# ==========================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .modern-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.25);
        position: relative;
        overflow: hidden;
    }
    
    .modern-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 50%;
    }
    
    .modern-header h1 {
        color: white;
        font-weight: 700;
        font-size: 2.2rem;
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .modern-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        font-weight: 300;
    }
    
    .badge-header {
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        padding: 0.4rem 1rem;
        border-radius: 50px;
        color: white;
        font-size: 0.8rem;
        display: inline-block;
        margin-top: 0.8rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stat-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
        border: 1px solid rgba(0, 0, 0, 0.04);
        transition: all 0.2s ease;
        height: 100%;
    }
    
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
    }
    
    .stat-card .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a1a2e;
        margin: 0.3rem 0;
    }
    
    .stat-card .stat-label {
        font-size: 0.85rem;
        color: #6b7280;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    
    .upload-area {
        border: 2px dashed #d1d5db;
        border-radius: 16px;
        padding: 3rem 2rem;
        text-align: center;
        background: #fafbfc;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .upload-area:hover {
        border-color: #667eea;
        background: #f8f9ff;
    }
    
    .upload-area .upload-icon {
        font-size: 3.5rem;
        margin-bottom: 0.5rem;
    }
    
    .result-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
        border: 1px solid rgba(0, 0, 0, 0.04);
        transition: all 0.3s ease;
    }
    
    .result-card:hover {
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
    }
    
    .birads-badge {
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    .birads-1 { background: #d1fae5; color: #065f46; }
    .birads-2 { background: #dbeafe; color: #1e40af; }
    .birads-3 { background: #fef3c7; color: #92400e; }
    .birads-4 { background: #fee2e2; color: #991b1b; }
    
    .custom-progress {
        background: #e5e7eb;
        border-radius: 50px;
        height: 8px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .custom-progress-bar {
        height: 100%;
        border-radius: 50px;
        transition: width 0.5s ease;
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeInUp 0.5s ease forwards;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# CONFIGURATION DES MODÈLES
# ==========================================================

IMG_SIZE_DENSITY = 224

CLASS_NAMES_DENSITY = {
    0: "ACR_A",
    1: "ACR_B", 
    2: "ACR_C",
    3: "ACR_D"
}

DENSITY_COLORS = {
    "ACR_A": "#10b981",
    "ACR_B": "#3b82f6", 
    "ACR_C": "#f59e0b",
    "ACR_D": "#ef4444"
}

DENSITY_DESC = {
    "ACR_A": "Densité graisseuse (<25% glandulaire)",
    "ACR_B": "Densité moyenne faible (25-50% glandulaire)",
    "ACR_C": "Densité moyenne élevée (50-75% glandulaire)",
    "ACR_D": "Densité extrême (>75% glandulaire)"
}

DENSITY_RISK = {
    "ACR_A": "Risque faible",
    "ACR_B": "Risque moyen",
    "ACR_C": "Risque modéré",
    "ACR_D": "Risque élevé"
}

MODELE_YOLO_PATH = "detecteur_masscalcif.pt"
IMGSZ = 1024

BIRADS_RECO = {
    "BI-RADS 1": "✅ Mammographie normale — aucune lésion détectée.",
    "BI-RADS 2": "🟢 Anomalie bénigne — surveillance standard.",
    "BI-RADS 3": "🟡 Anomalie probablement bénigne — surveillance court terme.",
    "BI-RADS 4": "🔴 Anomalie suspecte — biopsie recommandée.",
}

COULEURS_BIRADS = {
    "BI-RADS 1": "#10b981",
    "BI-RADS 2": "#3b82f6",
    "BI-RADS 3": "#f59e0b",
    "BI-RADS 4": "#ef4444",
}

# ==========================================================
# LOSS ORDINALE
# ==========================================================

@tf.keras.utils.register_keras_serializable(package="Custom")
def ordinal_loss(y_true, y_pred):
    weights = tf.constant([
        [0.0, 1.0, 2.0, 3.0],
        [1.0, 0.0, 1.0, 2.0],
        [2.0, 1.0, 0.0, 1.0],
        [3.0, 2.0, 1.0, 0.0],
    ], dtype=tf.float32)
    ce = tf.keras.losses.categorical_crossentropy(
        y_true, y_pred, label_smoothing=0.1
    )
    true_class = tf.cast(tf.argmax(y_true, axis=1), tf.int32)
    pred_class = tf.cast(tf.argmax(y_pred, axis=1), tf.int32)
    indices = tf.stack([true_class, pred_class], axis=1)
    penalty = tf.gather_nd(weights, indices)
    return ce + 0.5 * tf.cast(penalty, tf.float32)

# ==========================================================
# STRUCTURES DE DONNÉES
# ==========================================================

@dataclass
class Lesion:
    type_lesion: str
    birads: str
    details: dict = field(default_factory=dict)
    box: Optional[tuple] = None

@dataclass
class RapportLesions:
    fichier: str
    nb_masses: int
    nb_amas_calcif: int
    lesions: List[dict]
    birads_final: str
    recommandation: str

# ==========================================================
# FONCTIONS DE CONVERSION D'IMAGE UNIFORMISÉES
# ==========================================================

def image_to_array(img: Union[Image.Image, np.ndarray, str]) -> np.ndarray:
    """
    Convertit n'importe quel format d'image en array numpy (niveaux de gris).
    Supporte : PIL.Image, numpy array, chemin de fichier.
    """
    if isinstance(img, Image.Image):
        return np.array(img.convert("L"))
    elif isinstance(img, np.ndarray):
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img
    elif isinstance(img, str):
        return np.array(Image.open(img).convert("L"))
    else:
        raise ValueError(f"Type d'image non supporté: {type(img)}")

def image_to_rgb_array(img: Union[Image.Image, np.ndarray, str]) -> np.ndarray:
    """
    Convertit n'importe quel format d'image en array numpy RGB.
    """
    if isinstance(img, Image.Image):
        return np.array(img.convert("RGB"))
    elif isinstance(img, np.ndarray):
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 3:
            return img
        elif len(img.shape) == 3 and img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        return img
    elif isinstance(img, str):
        return np.array(Image.open(img).convert("RGB"))
    else:
        raise ValueError(f"Type d'image non supporté: {type(img)}")

# ==========================================================
# PRÉTRAITEMENT
# ==========================================================

def preprocess_mammogram_image(img_array: np.ndarray, img_size: int = 512) -> np.ndarray:
    """
    Prétraite une image de mammographie.
    img_array doit être un array numpy en niveaux de gris.
    """
    if len(img_array.shape) == 3:
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img = img_array.copy()
    
    # Vérifier les dimensions minimales
    if img.shape[0] < 10 or img.shape[1] < 10:
        raise ValueError(f"Image trop petite: {img.shape}")
    
    # Uniformisation fond blanc/noir
    border_pixels = np.concatenate([
        img[0, :], img[-1, :],
        img[:, 0], img[:, -1]
    ])
    border_mean = np.mean(border_pixels)
    if border_mean > 127:
        img = cv2.bitwise_not(img)
    
    # Détection du sein
    _, thresh = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        margin = 20
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(img.shape[1] - x, w + 2 * margin)
        h = min(img.shape[0] - y, h + 2 * margin)
        if w > 0 and h > 0:
            img = img[y:y+h, x:x+w]
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    img = clahe.apply(img)
    
    # Sharpening
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=2)
    img = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
    
    # Normalisation
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    
    # Resize avec conservation des proportions
    h, w = img.shape
    scale = min(img_size / w, img_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Canvas
    canvas = np.zeros((img_size, img_size), dtype=np.uint8)
    x_offset = (img_size - new_w) // 2
    y_offset = (img_size - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img
    
    return canvas

def prepare_for_model(img_array: np.ndarray, target_size: int) -> np.ndarray:
    """Prépare l'image pour le modèle CNN."""
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    img_resized = cv2.resize(img_rgb, (target_size, target_size))
    arr = img_resized.astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ==========================================================
# FONCTIONS DE DÉTECTION DES LÉSIONS
# ==========================================================

def masque_sein(img_gray):
    _, m = cv2.threshold(img_gray, 15, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=2)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.ones_like(img_gray) * 255
    c = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(img_gray)
    cv2.drawContours(mask, [c], -1, 255, -1)
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
    return cv2.erode(mask, k2, iterations=2)

def features_contour(contour):
    aire = cv2.contourArea(contour)
    perim = cv2.arcLength(contour, True)
    if perim == 0 or aire == 0:
        return {}
    circularite = 4 * np.pi * aire / (perim ** 2)
    hull = cv2.convexHull(contour)
    ah = cv2.contourArea(hull)
    convexite = aire / ah if ah > 0 else 0
    deficit_convexite = (ah - aire) / ah if ah > 0 else 0
    rapport_axes = 1.0
    if len(contour) >= 5:
        try:
            e = cv2.fitEllipse(contour)
            a, b = max(e[1]), min(e[1])
            rapport_axes = a / b if b > 0 else 1
        except:
            pass
    approx = cv2.approxPolyDP(contour, 0.005 * perim, True)
    rugosite = len(contour) / max(len(approx), 1)
    return {
        "circularite": circularite,
        "convexite": convexite,
        "deficit_convexite": deficit_convexite,
        "rapport_axes": rapport_axes,
        "rugosite": rugosite
    }

def classer_masse(features):
    if not features:
        return "irreguliere"
    if features["circularite"] >= 0.80 and features["rapport_axes"] < 1.4:
        return "ronde_arrondie"
    elif features["circularite"] >= 0.55 and features["rapport_axes"] < 2.5 and features["convexite"] > 0.80:
        return "ovale_elliptique"
    return "irreguliere"

def analyser_masse(crop):
    blur = cv2.GaussianBlur(crop, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cont, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cont:
        return {"forme": "indistinct", "score": 0.3}
    c = max(cont, key=cv2.contourArea)
    feat = features_contour(c)
    forme = classer_masse(feat)
    score = 0.1 if forme == "ronde_arrondie" else 0.3 if forme == "ovale_elliptique" else 0.5
    return {"forme": forme, "score": round(min(score, 1.0), 2)}

def detecter_masses(img_gray, model, conf):
    img_3c = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    res = model.predict(img_3c, conf=conf, iou=0.5, imgsz=IMGSZ, verbose=False)[0]
    masses = []
    if res.boxes is not None:
        for box in res.boxes:
            if int(box.cls[0]) != 0:
                continue
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            crop = img_gray[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            analyse = analyser_masse(crop)
            score = analyse["score"] + (1 - float(box.conf[0])) * 0.1
            if score >= 0.45:
                birads = "BI-RADS 4"
            elif score >= 0.25:
                birads = "BI-RADS 3"
            else:
                birads = "BI-RADS 2"
            masses.append(Lesion(
                "Masse",
                birads,
                {"forme": analyse["forme"], "confiance": round(float(box.conf[0]), 2), "score": score},
                (x1, y1, x2, y2)
            ))
    return masses

def detecter_calcifications(img_gray):
    mask = masque_sein(img_gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enh = clahe.apply(img_gray)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    tophat = cv2.morphologyEx(enh, cv2.MORPH_TOPHAT, se)
    zone = tophat[mask > 0]
    seuil = np.percentile(zone, 99) if zone.size else 30
    _, thr = cv2.threshold(tophat, seuil, 255, cv2.THRESH_BINARY)
    thr = cv2.bitwise_and(thr, thr, mask=mask)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(thr, cv2.MORPH_OPEN, k, iterations=1)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = []
    for c in contours:
        aire = cv2.contourArea(c)
        if 2 <= aire <= 90:
            M = cv2.moments(c)
            if M["m00"] > 0:
                points.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))
    return points

def regrouper_amas(points, img_w):
    if len(points) < 3:
        return []
    pts = np.array(points)
    rayon = 0.06 * img_w
    indices = list(range(len(pts)))
    amas = []
    while indices:
        base = indices.pop(0)
        groupe = [base]
        i = 0
        while i < len(indices):
            idx = indices[i]
            if any(np.linalg.norm(pts[idx] - pts[g]) < rayon for g in groupe):
                groupe.append(idx)
                indices.pop(i)
                i = 0
            else:
                i += 1
        if len(groupe) >= 3:
            amas.append([tuple(pts[g]) for g in groupe])
    return amas

def analyser_lesions(img_gray, model, conf):
    H, W = img_gray.shape
    masses = detecter_masses(img_gray, model, conf)
    points = detecter_calcifications(img_gray)
    amas = regrouper_amas(points, W)
    lesions = list(masses)
    for amas_pts in amas:
        xs = [p[0] for p in amas_pts]
        ys = [p[1] for p in amas_pts]
        box = (int(min(xs)) - 10, int(min(ys)) - 10, int(max(xs)) + 10, int(max(ys)) + 10)
        nb_calcif = len(amas_pts)
        if nb_calcif >= 10:
            birads = "BI-RADS 3"
        elif nb_calcif >= 3:
            birads = "BI-RADS 2"
        else:
            birads = "BI-RADS 2"
        lesions.append(Lesion(
            "Calcifications",
            birads,
            {"nombre": nb_calcif, "taille": "micro"},
            box
        ))
    if not lesions:
        birads_final = "BI-RADS 1"
    else:
        niveaux = {"BI-RADS 1": 0, "BI-RADS 2": 2, "BI-RADS 3": 3, "BI-RADS 4": 4}
        max_niveau = max(niveaux.get(l.birads, 0) for l in lesions)
        birads_final = f"BI-RADS {max_niveau}" if max_niveau > 0 else "BI-RADS 1"
    return RapportLesions(
        "image",
        len(masses),
        len(amas),
        [asdict(l) for l in lesions],
        birads_final,
        BIRADS_RECO.get(birads_final, "")
    )

def annoter_image_lesions(img_gray, rapport):
    img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    couleurs = {"BI-RADS 2": (0, 180, 0), "BI-RADS 3": (0, 180, 255), "BI-RADS 4": (0, 0, 220)}
    for l in rapport.lesions:
        if not l.get("box"):
            continue
        x1, y1, x2, y2 = l["box"]
        c = couleurs.get(l["birads"], (200, 200, 200))
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 3)
        label = f"{l['type_lesion']} {l['birads']}"
        cv2.putText(img, label, (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ==========================================================
# CHARGEMENT DES MODÈLES
# ==========================================================

@st.cache_resource
def load_density_model():
    try:
        model = tf.keras.models.load_model(
            "modele_densite_final.keras",
            compile=False
        )
        return model, None
    except Exception as e:
        return None, str(e)

@st.cache_resource
def load_yolo_model():
    try:
        return YOLO(MODELE_YOLO_PATH), None
    except Exception as e:
        return None, str(e)

# ==========================================================
# PRÉDICTION DENSITÉ
# ==========================================================

def predict_density(model, img_array):
    probs = model.predict(img_array, verbose=0)[0]
    idx = int(np.argmax(probs))
    label = CLASS_NAMES_DENSITY[idx]
    all_probs = {CLASS_NAMES_DENSITY[i]: float(probs[i]) for i in range(4)}
    return label, float(probs[idx]), all_probs

# ==========================================================
# ANALYSE COMPLÈTE - VERSION CORRIGÉE
# ==========================================================

def analyze_complete(img, density_model, yolo_model, conf_threshold=0.15):
    """
    Analyse complète d'une image.
    Supporte tous les formats d'image sans discrimination.
    """
    results = {"success": True, "filename": getattr(img, 'name', 'image')}
    
    try:
        # Conversion uniforme de l'image
        img_gray = image_to_array(img)
        
        # Vérification des dimensions
        if img_gray.shape[0] < 10 or img_gray.shape[1] < 10:
            raise ValueError(f"Image trop petite: {img_gray.shape}")
        
        # Prétraitement pour la densité
        img_preprocessed = preprocess_mammogram_image(img_gray, 512)
        results["preprocessed"] = img_preprocessed
        
        # Analyse de densité
        arr_density = prepare_for_model(img_preprocessed, IMG_SIZE_DENSITY)
        label, conf, probs = predict_density(density_model, arr_density)
        
        results.update({
            "density_label": label,
            "density_confidence": conf,
            "density_probs": probs,
            "density_risk": DENSITY_RISK.get(label, ""),
            "density_desc": DENSITY_DESC.get(label, "")
        })
        
        # Analyse des lésions (si YOLO disponible)
        if yolo_model is not None:
            rapport = analyser_lesions(img_gray, yolo_model, conf_threshold)
            results["lesions"] = rapport
            img_annotated = annoter_image_lesions(img_gray, rapport)
            results["annotated"] = img_annotated
            results["birads"] = rapport.birads_final
            results["recommandation"] = rapport.recommandation
            results["nb_masses"] = rapport.nb_masses
            results["nb_calcifications"] = rapport.nb_amas_calcif

    except Exception as e:
        results = {"success": False, "error": str(e), "filename": getattr(img, 'name', 'image')}
    
    return results

# ==========================================================
# INTERFACE PRINCIPALE
# ==========================================================

def main():
    # Header
    st.markdown("""
    <div class="modern-header">
        <h1>🏥 MammoAI</h1>
        <p>Analyse avancée de mammographies — Densité mammaire ACR + Détection des lésions BI-RADS</p>
        <div class="badge-header">
            <span>⚡ IA médicale · Précision diagnostique</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Chargement des modèles
    with st.spinner("🔄 Chargement des modèles d'intelligence artificielle..."):
        density_model, density_error = load_density_model()
        yolo_model, yolo_error = load_yolo_model()
    
    # Stats des modèles
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">🧠 Modèle Densité</div>
            <div class="stat-value">✅</div>
            <div style="font-size:0.8rem;color:#6b7280;">ACR A/B/C/D</div>
        </div>
        """ if density_model else """
        <div class="stat-card">
            <div class="stat-label">🧠 Modèle Densité</div>
            <div class="stat-value">❌</div>
            <div style="font-size:0.8rem;color:#ef4444;">Non chargé</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">🎯 Modèle Lésions</div>
            <div class="stat-value">✅</div>
            <div style="font-size:0.8rem;color:#6b7280;">YOLO · BI-RADS</div>
        </div>
        """ if yolo_model else """
        <div class="stat-card">
            <div class="stat-label">🎯 Modèle Lésions</div>
            <div class="stat-value">⚠️</div>
            <div style="font-size:0.8rem;color:#f59e0b;">Optionnel</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">📊 Version</div>
            <div class="stat-value">v2.1</div>
            <div style="font-size:0.8rem;color:#6b7280;">TensorFlow {tf.__version__}</div>
        </div>
        """, unsafe_allow_html=True)
    
    if density_model is None:
        st.error("❌ Le modèle de densité est requis. Vérifiez que 'modele_densite_final.keras' est présent.")
        st.stop()
    
    st.markdown("---")
    
    # Zone d'upload
    uploaded_files = st.file_uploader(
        "📤 Importer des mammographies",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
        accept_multiple_files=True
    )
    
    if not uploaded_files:
        st.markdown("""
        <div class="upload-area">
            <div class="upload-icon">🖼️</div>
            <h3 style="margin:0.5rem 0;color:#1a1a2e;">Déposez vos images ici</h3>
            <p style="color:#6b7280;margin:0;">Formats supportés : JPG, PNG, BMP, TIFF, WEBP</p>
            <p style="color:#9ca3af;font-size:0.85rem;margin-top:0.5rem;">
                Aucune discrimination de format — toutes les images sont traitées uniformément
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📖 Guide d'utilisation"):
            st.markdown("""
            ### Comment utiliser MammoAI ?
            
            1. **Importez** vos mammographies via le bouton ci-dessus
            2. **Analysez** en cliquant sur "Lancer l'analyse"
            3. **Consultez** les résultats : densité ACR + détection des lésions
            
            ### Interprétation des résultats
            
            **Densité mammaire (ACR) :**
            - 🟢 **ACR A** : Densité graisseuse (<25%) — Risque faible
            - 🔵 **ACR B** : Densité moyenne faible (25-50%) — Risque moyen
            - 🟡 **ACR C** : Densité moyenne élevée (50-75%) — Risque modéré
            - 🔴 **ACR D** : Densité extrême (>75%) — Risque élevé
            
            **Classification BI-RADS :**
            - **BI-RADS 1** : Normal
            - **BI-RADS 2** : Bénin
            - **BI-RADS 3** : Probablement bénin
            - **BI-RADS 4** : Suspect
            """)
        return
    
    # Barre d'action
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.success(f"📁 {len(uploaded_files)} mammographie(s) importée(s)")
    
    with col2:
        analyze_btn = st.button("🚀 Lancer l'analyse", type="primary", use_container_width=True)
    
    with col3:
        export_btn = st.button("📊 Exporter CSV", use_container_width=True)
    
    with col4:
        clear_btn = st.button("🗑️ Effacer", use_container_width=True)
        if clear_btn:
            st.session_state.results_complete = {}
            st.rerun()
    
    # Paramètres
    with st.expander("⚙️ Paramètres avancés"):
        col1, col2 = st.columns(2)
        with col1:
            conf_threshold = st.slider(
                "🎯 Seuil de confiance YOLO",
                min_value=0.05,
                max_value=0.5,
                value=0.15,
                step=0.05,
                help="Plus le seuil est bas, plus de détections sont faites"
            )
        with col2:
            st.caption("💡 Recommandation : 0.15 pour une sensibilité maximale")
    
    # Initialisation des résultats
    if "results_complete" not in st.session_state:
        st.session_state.results_complete = {}
    
    # Analyse
    if analyze_btn:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, f in enumerate(uploaded_files):
            status_text.markdown(f"**Analyse en cours :** `{f.name}` ({i+1}/{len(uploaded_files)})")
            img = Image.open(f)
            st.session_state.results_complete[f.name] = analyze_complete(
                img, density_model, yolo_model, conf_threshold
            )
            progress_bar.progress((i + 1) / len(uploaded_files))
            time.sleep(0.1)
        
        progress_bar.empty()
        status_text.empty()
        st.balloons()
        st.success("✅ Analyse terminée avec succès !")
    
    # Export CSV
    if export_btn and st.session_state.results_complete:
        rows = []
        for fname, res in st.session_state.results_complete.items():
            row = {"Fichier": fname}
            if res.get("success"):
                row["Statut"] = "OK"
                row["Densité ACR"] = res.get("density_label", "—")
                row["Confiance"] = f"{res.get('density_confidence', 0)*100:.1f}%"
                row["Risque"] = res.get("density_risk", "—")
                row["BI-RADS"] = res.get("birads", "—")
                row["Masses"] = res.get("nb_masses", 0)
                row["Calcifications"] = res.get("nb_calcifications", 0)
            else:
                row["Statut"] = "Erreur"
                row["Erreur"] = res.get("error", "")
            rows.append(row)
        
        csv = pd.DataFrame(rows).to_csv(index=False)
        st.download_button(
            "📥 Télécharger le rapport CSV",
            data=csv,
            file_name="mammoai_rapport.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # ==========================================================
    # AFFICHAGE DES RÉSULTATS
    # ==========================================================
    
    if st.session_state.results_complete:
        st.markdown("---")
        st.markdown("## 📊 Résultats de l'analyse")
        
        view_mode = st.radio(
            "Mode d'affichage",
            ["🔲 Grille", "📋 Détail", "📊 Tableau"],
            horizontal=True
        )
        
        def density_badge_html(res):
            label = res.get("density_label", "—")
            desc = res.get("density_desc", "")
            conf = res.get("density_confidence", 0) * 100
            risk = res.get("density_risk", "")
            color = DENSITY_COLORS.get(label, "#6b7280")
            
            return f"""
            <div style="background:white;border-radius:12px;padding:1rem;border-left:4px solid {color};">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-weight:600;font-size:1.1rem;color:{color};">{label}</span>
                    <span style="font-size:0.8rem;color:#6b7280;">{conf:.1f}%</span>
                </div>
                <div style="font-size:0.85rem;color:#4b5563;margin:0.2rem 0;">{desc}</div>
                <div style="font-size:0.8rem;color:{color};">{risk}</div>
                <div class="custom-progress" style="margin-top:0.5rem;">
                    <div class="custom-progress-bar" style="width:{conf}%;background:{color};"></div>
                </div>
            </div>
            """
        
        def birads_badge_html(birads):
            if birads == "BI-RADS 1":
                return f'<span class="birads-badge birads-1">✅ {birads}</span>'
            elif birads == "BI-RADS 2":
                return f'<span class="birads-badge birads-2">🟢 {birads}</span>'
            elif birads == "BI-RADS 3":
                return f'<span class="birads-badge birads-3">🟡 {birads}</span>'
            elif birads == "BI-RADS 4":
                return f'<span class="birads-badge birads-4">🔴 {birads}</span>'
            return f'<span class="birads-badge" style="background:#e5e7eb;">{birads}</span>'
        
        # ---------- GRILLE ----------
        if view_mode == "🔲 Grille":
            cols = st.columns(3)
            for idx, f in enumerate(uploaded_files):
                with cols[idx % 3]:
                    res = st.session_state.results_complete.get(f.name)
                    
                    st.markdown(f"""
                    <div class="result-card fade-in">
                        <div style="font-weight:600;margin-bottom:0.5rem;font-size:0.9rem;color:#374151;">
                            📷 {f.name[:20]}{'...' if len(f.name) > 20 else ''}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if res and res.get("success"):
                        img = Image.open(f)
                        st.image(img, use_container_width=True)
                        st.markdown(density_badge_html(res), unsafe_allow_html=True)
                        
                        birads = res.get("birads", "—")
                        st.markdown(f"""
                        <div style="margin-top:0.5rem;display:flex;justify-content:space-between;align-items:center;">
                            <span style="font-weight:500;color:#374151;">BI-RADS</span>
                            {birads_badge_html(birads)}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.caption(f"🔹 {res.get('nb_masses', 0)} masses · 🔸 {res.get('nb_calcifications', 0)} calcifications")
                    elif res:
                        st.error(f"❌ {res.get('error', 'Erreur')[:50]}...")
                    else:
                        st.info("⏳ Non analysé")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
        
        # ---------- DÉTAIL ----------
        elif view_mode == "📋 Détail":
            for f in uploaded_files:
                res = st.session_state.results_complete.get(f.name)
                if not res or not res.get("success"):
                    continue
                
                with st.expander(f"📷 {f.name}", expanded=True):
                    col_img, col_info = st.columns([1, 1])
                    
                    with col_img:
                        img = Image.open(f)
                        st.image(img, use_container_width=True)
                        
                        if res.get("annotated") is not None:
                            st.caption("🔬 Détection des lésions")
                            st.image(res["annotated"], use_container_width=True)
                    
                    with col_info:
                        st.markdown("### 🧬 Densité mammaire")
                        st.markdown(density_badge_html(res), unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        st.markdown("### 🎯 Classification BI-RADS")
                        birads = res.get("birads", "—")
                        st.markdown(f"""
                        <div style="display:flex;align-items:center;gap:1rem;padding:0.5rem 0;">
                            {birads_badge_html(birads)}
                            <span style="font-size:0.9rem;color:#4b5563;">{res.get('recommandation', '')}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Masses", res.get("nb_masses", 0))
                        with col2:
                            st.metric("Calcifications", res.get("nb_calcifications", 0))
                        
                        lesions_data = res.get("lesions")
                        if lesions_data and lesions_data.lesions:
                            st.markdown("---")
                            st.markdown("### 📋 Détail des lésions")
                            data = []
                            for i, l in enumerate(lesions_data.lesions, 1):
                                data.append({
                                    "#": i,
                                    "Type": l["type_lesion"],
                                    "BI-RADS": l["birads"],
                                    "Détails": str(l["details"])[:60] + ("..." if len(str(l["details"])) > 60 else "")
                                })
                            st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
        
        # ---------- TABLEAU ----------
        else:
            rows = []
            for f in uploaded_files:
                res = st.session_state.results_complete.get(f.name, {})
                if res.get("success"):
                    rows.append({
                        "Fichier": f.name[:30] + ("..." if len(f.name) > 30 else ""),
                        "Densité ACR": res.get("density_label", "—"),
                        "Confiance": f"{res.get('density_confidence', 0)*100:.1f}%",
                        "Risque": res.get("density_risk", "—"),
                        "BI-RADS": res.get("birads", "—"),
                        "Masses": res.get("nb_masses", 0),
                        "Calcif.": res.get("nb_calcifications", 0),
                    })
            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Densité ACR": st.column_config.TextColumn("Densité ACR", width="small"),
                        "BI-RADS": st.column_config.TextColumn("BI-RADS", width="small"),
                    }
                )

if __name__ == "__main__":
    main()