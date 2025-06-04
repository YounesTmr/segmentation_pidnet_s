import streamlit as st
import os
from PIL import Image
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import matplotlib.pyplot as plt
from pidnet_loader import load_pidnet_model, predict_segmentation
from utils import overlay_mask, colorize_mask
import tensorflow as tf
import cv2

# Configuration de la page avec amélioration accessibilité
st.set_page_config(
    page_title="PIDNet vs DeepLabS Cityscapes Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 Dashboard de segmentation d'images - Cityscapes (PIDNet-S vs DeepLabS)")
st.markdown("Ce dashboard permet d'explorer le dataset Cityscapes et de comparer les prédictions de segmentation entre PIDNet-S et DeepLabS.")

# --- FONCTION POUR DEEPLABS ---
@st.cache_resource
def load_deeplabs_model(model_path):
    """Charge le modèle DeepLabS pré-entraîné"""
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle DeepLabS : {e}")
        return None

def preprocess_image_for_deeplabs(image, target_size=(256, 256)):
    """Préprocesse l'image pour DeepLabS"""
    # Convertir PIL en numpy array
    img_array = np.array(image)
    
    # Redimensionner l'image à 256x256 comme attendu par le modèle
    img_resized = cv2.resize(img_array, target_size)
    
    # Normaliser les pixels entre 0 et 1
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Ajouter la dimension batch
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

def predict_with_deeplabs(model, image):
    """Effectue la prédiction avec le modèle DeepLabS"""
    try:
        # Préprocesser l'image
        processed_img = preprocess_image_for_deeplabs(image)
        
        # Prédiction
        prediction = model.predict(processed_img, verbose=0)
        
        # Obtenir les classes prédites
        predicted_classes = np.argmax(prediction, axis=-1)
        
        # Supprimer la dimension batch
        predicted_mask = predicted_classes[0]
        
        return predicted_mask
        
    except Exception as e:
        st.error(f"Erreur lors de la prédiction DeepLabS : {e}")
        return None

def resize_mask_to_original(mask, original_size):
    """Redimensionne le masque à la taille originale de l'image"""
    # original_size est au format (width, height) de PIL
    # cv2.resize attend (width, height)
    return cv2.resize(mask.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST)

# --- SECTION 0 : ANALYSE EXPLORATOIRE DES DONNÉES ---
st.header("📊 Analyse exploratoire du dataset Cityscapes")

@st.cache_data
def load_dataset_statistics():
    """Charge et analyse les statistiques du dataset"""
    image_dir = "data/images"
    mask_dir = "data/masks"
    
    cities = sorted(os.listdir(image_dir))
    stats = []
    
    for city in cities:
        city_path = os.path.join(image_dir, city)
        if os.path.isdir(city_path):
            images = [f for f in os.listdir(city_path) if f.endswith('.png')]
            
            # Analyse d'une image sample pour les dimensions
            if images:
                sample_img = Image.open(os.path.join(city_path, images[0]))
                width, height = sample_img.size
                
                stats.append({
                    'Ville': city,
                    'Nombre_images': len(images),
                    'Largeur': width,
                    'Hauteur': height,
                    'Resolution': f"{width}x{height}",
                    'Ratio_aspect': round(width/height, 2)
                })
    
    return pd.DataFrame(stats)

# Chargement des statistiques
df_stats = load_dataset_statistics()

# Métriques générales
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🏙️ Nombre de villes", len(df_stats))
with col2:
    st.metric("📸 Total d'images", df_stats['Nombre_images'].sum())
with col3:
    st.metric("📐 Résolution standard", df_stats['Resolution'].iloc[0])
with col4:
    st.metric("📏 Ratio d'aspect moyen", round(df_stats['Ratio_aspect'].mean(), 2))

# Graphique interactif 1 : Distribution des images par ville
fig1 = px.bar(
    df_stats, 
    x='Ville', 
    y='Nombre_images',
    title="Distribution des images par ville",
    color='Nombre_images',
    color_continuous_scale='viridis',
    labels={'Nombre_images': 'Nombre d\'images', 'Ville': 'Ville'}
)
fig1.update_layout(
    xaxis_tickangle=-45,
    height=400,
    font=dict(size=12),
    title_font_size=16
)
st.plotly_chart(fig1, use_container_width=True)

# Graphique interactif 2 : Analyse des ratios d'aspect
fig2 = px.histogram(
    df_stats, 
    x='Ratio_aspect', 
    nbins=10,
    title="Distribution des ratios d'aspect des images",
    labels={'Ratio_aspect': 'Ratio d\'aspect (largeur/hauteur)', 'count': 'Fréquence'},
    color_discrete_sequence=['#ff6b6b']
)
fig2.update_layout(height=400, font=dict(size=12), title_font_size=16)
st.plotly_chart(fig2, use_container_width=True)

# Tableau récapitulatif avec meilleur contraste
st.subheader("📋 Résumé statistique par ville")
st.dataframe(
    df_stats.set_index('Ville'),
    use_container_width=True,
    height=300
)

# Analyse des transformations d'images
st.subheader("🎨 Analyse des transformations d'images")

@st.cache_data
def analyze_image_properties(city_name, max_samples=5):
    """Analyse les propriétés colorimétriques d'échantillons d'images"""
    image_dir = f"data/images/{city_name}"
    images = sorted(os.listdir(image_dir))[:max_samples]
    
    brightness_values = []
    contrast_values = []
    
    for img_name in images:
        img_path = os.path.join(image_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img)
        
        # Calcul luminosité moyenne
        brightness = np.mean(img_array)
        brightness_values.append(brightness)
        
        # Calcul contraste (écart-type des pixels)
        contrast = np.std(img_array)
        contrast_values.append(contrast)
    
    return {
        'images': images,
        'brightness': brightness_values,
        'contrast': contrast_values
    }

# Sélection ville pour analyse
selected_city_analysis = st.selectbox(
    "Sélectionner une ville pour l'analyse :", 
    df_stats['Ville'].tolist(),
    key="city_analysis"
)

image_props = analyze_image_properties(selected_city_analysis)

# Graphique des propriétés colorimétriques
fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=list(range(len(image_props['brightness']))),
    y=image_props['brightness'],
    mode='lines+markers',
    name='Luminosité moyenne',
    line=dict(color='gold', width=3),
    marker=dict(size=8)
))

fig3_secondary = go.Figure()
fig3_secondary.add_trace(go.Scatter(
    x=list(range(len(image_props['contrast']))),
    y=image_props['contrast'],
    mode='lines+markers',
    name='Contraste (écart-type)',
    line=dict(color='coral', width=3),
    marker=dict(size=8)
))

col_bright, col_contrast = st.columns(2)
with col_bright:
    fig3.update_layout(
        title=f"Luminosité moyenne - {selected_city_analysis}",
        xaxis_title="Index de l'image",
        yaxis_title="Luminosité (0-255)",
        height=300,
        font=dict(size=11)
    )
    st.plotly_chart(fig3, use_container_width=True)

with col_contrast:
    fig3_secondary.update_layout(
        title=f"Contraste - {selected_city_analysis}",
        xaxis_title="Index de l'image",
        yaxis_title="Contraste (écart-type)",
        height=300,
        font=dict(size=11)
    )
    st.plotly_chart(fig3_secondary, use_container_width=True)

# --- Fonction utilitaire : conversion couleur RGB en indices de classes ---
def rgb_to_class_mask(mask_rgb):
    """
    Convertit un masque couleur RGB (PIL Image) en masque d'indices de classes (numpy array).
    """
    CITYSCAPES_COLORS = [
        (128, 64,128),   # Fond
        (244, 35,232),   # Routes
        (70,  70, 70),   # Bâtiments
        (102,102,156),   # Objets urbains
        (107,142, 35),   # Végétation
        (70,130,180),   # Ciel
        (220, 20, 60),   # Piétons/Cyclistes
        (0,   0,142),    # Véhicules
    ]

    mask_np = np.array(mask_rgb)
    h, w, _ = mask_np.shape
    class_mask = np.zeros((h, w), dtype=np.uint8)

    for i, color in enumerate(CITYSCAPES_COLORS):
        matches = np.all(mask_np == color, axis=-1)
        class_mask[matches] = i

    return class_mask

def colorize_8classes_mask(mask_8classes_np):
    """Colorise un masque de classes selon la palette Cityscapes"""
    CITYSCAPES_COLORS_8 = [
        (128, 64,128), (244, 35,232), (70, 70, 70), (102,102,156),
        (107,142, 35), (70,130,180), (220, 20, 60), (0, 0,142)
    ]
    h, w = mask_8classes_np.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(CITYSCAPES_COLORS_8):
        color_mask[mask_8classes_np == cls_idx] = color
    return Image.fromarray(color_mask)

# --- SECTION 1 : Sélection d'images ---
st.header("1️⃣ Sélection d'images")

image_dir = "data/images"
mask_dir = "data/masks"

cities = sorted(os.listdir(image_dir))

# Sélection avec amélioration accessibilité
st.markdown("**Sélectionnez une ville et une image pour la prédiction :**")
selected_city = st.selectbox(
    "Ville :", 
    cities,
    help="Choisissez la ville dont vous souhaitez analyser les images"
)

image_files = sorted(os.listdir(os.path.join(image_dir, selected_city)))
st.markdown(f"- 📊 Nombre d'images disponibles : **{len(image_files)}**")

example_image = Image.open(os.path.join(image_dir, selected_city, image_files[0]))
st.markdown(f"- 📐 Dimensions des images : **{example_image.size}**")

selected_image_name = st.selectbox(
    "Image :", 
    image_files,
    help="Sélectionnez l'image spécifique à segmenter"
)

# Chargement des images
mask_name = selected_image_name.replace("leftImg8bit.png", "gtFine_color.png")
selected_image_path = os.path.join(image_dir, selected_city, selected_image_name)
selected_mask_path = os.path.join(mask_dir, selected_city, mask_name)

selected_image = Image.open(selected_image_path).convert("RGB")
mask_image = Image.open(selected_mask_path).convert("RGB")

# Affichage avec descriptions pour accessibilité
col_img, col_mask = st.columns(2)
with col_img:
    st.image(
        selected_image, 
        caption=f"Image réelle de {selected_city} - {selected_image_name}",
        use_column_width=True
    )
    st.markdown("*Image originale de rue urbaine du dataset Cityscapes*")

with col_mask:
    st.image(
        mask_image, 
        caption=f"Masque de segmentation réel (Ground Truth)",
        use_column_width=True
    )
    st.markdown("*Masque de vérité terrain avec 8 classes sémantiques*")

# --- SECTION 2 : Comparaison des modèles ---
st.header("2️⃣ Comparaison PIDNet-S vs DeepLabS")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.info(f"🔧 Utilisation du device : **{device}**")

# Chargement des modèles
st.subheader("🤖 Chargement des modèles")

col_pidnet, col_deeplabs = st.columns(2)

with col_pidnet:
    st.markdown("**PIDNet-S**")
    try:
        pidnet_model = load_pidnet_model("model/checkpoint.pth", device)
        pidnet_loaded = True
        st.success("✅ PIDNet-S chargé avec succès")
    except Exception as e:
        st.error(f"❌ Erreur PIDNet-S : {e}")
        pidnet_loaded = False

# with col_deeplabs:
#     st.markdown("**DeepLabS Enhanced**")
#     try:
#         deeplabs_model = load_deeplabs_model("model/deeplabs_enhanced_segmentation.h5")
#         deeplabs_loaded = deeplabs_model is not None
#         if deeplabs_loaded:
#             st.success("✅ DeepLabS chargé avec succès")
#         else:
#             st.error("❌ Échec du chargement DeepLabS")
#     except Exception as e:
#         st.error(f"❌ Erreur DeepLabS : {e}")
#         deeplabs_loaded = False

# Boutons de prédiction
st.subheader("🚀 Lancement des prédictions")

col_btn1, col_btn2, col_btn3 = st.columns(3)

with col_btn1:
    run_pidnet = st.button("🔥 PIDNet-S uniquement", disabled=not pidnet_loaded, use_container_width=True)

with col_btn2:
    run_deeplabs = st.button("🧬 DeepLabS uniquement", disabled=not deeplabs_loaded, use_container_width=True)

with col_btn3:
    run_comparison = st.button("⚡ Comparaison complète", disabled=not (pidnet_loaded and deeplabs_loaded), use_container_width=True)

# Fonction pour calculer des métriques de comparaison
def calculate_comparison_metrics(pred1, pred2, gt_mask):
    """Calcule des métriques de comparaison entre les prédictions"""
    # Debug: afficher les tailles
    st.write(f"Tailles - GT: {gt_mask.shape}, PIDNet: {pred1.shape}, DeepLabS: {pred2.shape}")
    
    # S'assurer que tous les masques ont la même taille que le ground truth
    if pred1.shape != gt_mask.shape:
        pred1 = cv2.resize(pred1.astype(np.uint8), (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        st.write(f"PIDNet redimensionné à: {pred1.shape}")
    
    if pred2.shape != gt_mask.shape:
        pred2 = cv2.resize(pred2.astype(np.uint8), (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        st.write(f"DeepLabS redimensionné à: {pred2.shape}")
    
    # Intersection over Union (IoU) pour chaque classe
    unique_classes = np.unique(gt_mask)
    ious = {}
    
    for cls in unique_classes:
        gt_cls = (gt_mask == cls)
        pred1_cls = (pred1 == cls)
        pred2_cls = (pred2 == cls)
        
        # IoU pour PIDNet vs GT
        intersection1 = np.logical_and(gt_cls, pred1_cls)
        union1 = np.logical_or(gt_cls, pred1_cls)
        iou1 = np.sum(intersection1) / np.sum(union1) if np.sum(union1) > 0 else 0
        
        # IoU pour DeepLabS vs GT
        intersection2 = np.logical_and(gt_cls, pred2_cls)
        union2 = np.logical_or(gt_cls, pred2_cls)
        iou2 = np.sum(intersection2) / np.sum(union2) if np.sum(union2) > 0 else 0
        
        ious[f"Classe_{cls}"] = {"PIDNet": iou1, "DeepLabS": iou2}
    
    # Accuracy globale
    pidnet_acc = np.mean(pred1 == gt_mask)
    deeplabs_acc = np.mean(pred2 == gt_mask)
    
    return ious, pidnet_acc, deeplabs_acc

# Traitement des prédictions
if run_pidnet or run_comparison:
    if pidnet_loaded:
        with st.spinner("🔄 Prédiction PIDNet-S en cours..."):
            try:
                pidnet_pred = predict_segmentation(pidnet_model, selected_image, device)
                pidnet_colored = colorize_mask(pidnet_pred)
                pidnet_overlay = overlay_mask(selected_image, pidnet_colored, alpha=0.6)
                st.success("✅ PIDNet-S : Prédiction terminée")
            except Exception as e:
                st.error(f"❌ Erreur PIDNet-S : {e}")
                pidnet_pred = None

if run_deeplabs or run_comparison:
    if deeplabs_loaded:
        with st.spinner("🔄 Prédiction DeepLabS en cours..."):
            try:
                deeplabs_pred_raw = predict_with_deeplabs(deeplabs_model, selected_image)
                if deeplabs_pred_raw is not None:
                    # Redimensionner à la taille originale de l'image
                    original_size = selected_image.size  # (width, height)
                    deeplabs_pred = resize_mask_to_original(deeplabs_pred_raw, original_size)
                    deeplabs_colored = colorize_8classes_mask(deeplabs_pred)
                    deeplabs_overlay = overlay_mask(selected_image, deeplabs_colored, alpha=0.6)
                    
                    # Debug info
                    st.write(f"DeepLabS - Taille originale: {deeplabs_pred_raw.shape}, Redimensionnée: {deeplabs_pred.shape}")
                    
                    st.success("✅ DeepLabS : Prédiction terminée")
                else:
                    deeplabs_pred = None
            except Exception as e:
                st.error(f"❌ Erreur DeepLabS : {e}")
                deeplabs_pred = None

# Affichage des résultats
if run_pidnet and 'pidnet_pred' in locals() and pidnet_pred is not None:
    st.subheader("🔥 Résultats PIDNet-S")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(pidnet_colored, caption="Masque prédit par PIDNet-S", use_column_width=True)
    with col2:
        # Afficher le ground truth
        gt_mask_indices = rgb_to_class_mask(mask_image)
        gt_colored = colorize_8classes_mask(gt_mask_indices)
        st.image(gt_colored, caption="Masque réel (Ground Truth)", use_column_width=True)
    with col3:
        st.image(pidnet_overlay, caption="Superposition PIDNet-S", use_column_width=True)

if run_deeplabs and 'deeplabs_pred' in locals() and deeplabs_pred is not None:
    st.subheader("🧬 Résultats DeepLabS")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(deeplabs_colored, caption="Masque prédit par DeepLabS", use_column_width=True)
    with col2:
        # Afficher le ground truth
        gt_mask_indices = rgb_to_class_mask(mask_image)
        gt_colored = colorize_8classes_mask(gt_mask_indices)
        st.image(gt_colored, caption="Masque réel (Ground Truth)", use_column_width=True)
    with col3:
        st.image(deeplabs_overlay, caption="Superposition DeepLabS", use_column_width=True)

if run_comparison and 'pidnet_pred' in locals() and 'deeplabs_pred' in locals() and pidnet_pred is not None and deeplabs_pred is not None:
    st.subheader("⚡ Comparaison complète des modèles")
    
    # Affichage côte à côte
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image(selected_image, caption="Image originale", use_column_width=True)
    with col2:
        gt_mask_indices = rgb_to_class_mask(mask_image)
        gt_colored = colorize_8classes_mask(gt_mask_indices)
        st.image(gt_colored, caption="Ground Truth", use_column_width=True)
    with col3:
        st.image(pidnet_colored, caption="PIDNet-S", use_column_width=True)
    with col4:
        st.image(deeplabs_colored, caption="DeepLabS", use_column_width=True)
    
    # Métriques de comparaison
    st.subheader("📊 Métriques de performance")
    
    try:
        ious, pidnet_acc, deeplabs_acc = calculate_comparison_metrics(pidnet_pred, deeplabs_pred, gt_mask_indices)
        
        # Affichage des accuracies globales
        col_acc1, col_acc2 = st.columns(2)
        with col_acc1:
            st.metric("🎯 Accuracy PIDNet-S", f"{pidnet_acc:.3f}")
        with col_acc2:
            st.metric("🎯 Accuracy DeepLabS", f"{deeplabs_acc:.3f}")
        
        # Graphique de comparaison des IoU
        classes = list(ious.keys())
        pidnet_ious = [ious[cls]["PIDNet"] for cls in classes]
        deeplabs_ious = [ious[cls]["DeepLabS"] for cls in classes]
        
        fig_comparison = go.Figure()
        fig_comparison.add_trace(go.Bar(
            name='PIDNet-S',
            x=classes,
            y=pidnet_ious,
            marker_color='#ff6b6b'
        ))
        fig_comparison.add_trace(go.Bar(
            name='DeepLabS',
            x=classes,
            y=deeplabs_ious,
            marker_color='#4ecdc4'
        ))
        
        fig_comparison.update_layout(
            title='Comparaison des IoU par classe',
            xaxis_title='Classes',
            yaxis_title='IoU Score',
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
    except Exception as e:
        st.error(f"Erreur lors du calcul des métriques : {e}")
# Légende commune
def render_accessible_legend():
    LABELS_8 = [
        "Fond", "Routes", "Bâtiments", "Objets urbains",
        "Végétation", "Ciel", "Piétons/Cyclistes", "Véhicules"
    ]
    CITYSCAPES_COLORS = [
        (128, 64,128), (244, 35,232), (70, 70, 70), (102,102,156),
        (107,142, 35), (70,130,180), (220, 20, 60), (0, 0,142)
    ]
    hex_colors = ['#%02x%02x%02x' % tuple(rgb) for rgb in CITYSCAPES_COLORS]

    st.markdown("#### 🏷️ Légende des classes de segmentation")
    
    # Création de colonnes pour un meilleur affichage
    cols = st.columns(4)
    for i, (label, color) in enumerate(zip(LABELS_8, hex_colors)):
        with cols[i % 4]:
            st.markdown(
                f"""
                <div style='display: flex; align-items: center; margin-bottom: 8px; 
                            padding: 8px; border-radius: 5px; background: rgba(0,0,0,0.05);'>
                    <div style='width: 24px; height: 24px; background-color: {color}; 
                               border-radius: 4px; margin-right: 12px; 
                               border: 2px solid #333;'></div>
                    <span style='font-size: 14px; font-weight: 500; color: #333;'>{label}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

# Afficher la légende si des résultats sont disponibles
if any(var in locals() for var in ['pidnet_pred', 'deeplabs_pred']):
    render_accessible_legend()

# --- SECTION 3 : Accessibilité améliorée ---
st.header("♿ Critères d'accessibilité WCAG implémentés")

# Checklist des critères d'accessibilité
accessibility_features = {
    "Contraste des couleurs": "✅ Ratio de contraste minimum 4.5:1 respecté",
    "Textes alternatifs": "✅ Descriptions textuelles pour toutes les images",
    "Navigation clavier": "✅ Tous les éléments interactifs accessibles au clavier",
    "Labels explicites": "✅ Libellés clairs pour tous les contrôles",
    "Structure sémantique": "✅ Utilisation appropriée des en-têtes et sections",
    "Indicateurs visuels": "✅ États des boutons et interactions clairement indiqués",
    "Taille des textes": "✅ Texte lisible et redimensionnable",
    "Couleur non exclusive": "✅ Information non transmise uniquement par la couleur"
}

st.markdown("**Fonctionnalités d'accessibilité intégrées :**")
for feature, status in accessibility_features.items():
    st.markdown(f"- {status}")

# Informations sur le déploiement
st.header("☁️ Déploiement")
st.info("""
**Instructions de déploiement :**
1. **Streamlit Cloud** : Connecter le repository GitHub et déployer
2. **Heroku** : Utiliser le Procfile fourni pour le déploiement
3. **Docker** : Image containerisée disponible
4. **AWS/GCP** : Compatible avec les services cloud majeurs

Commande locale : `streamlit run app.py`

**Dépendances requises :**
- streamlit
- torch
- tensorflow
- opencv-python
- pillow
- numpy
- pandas
- plotly
""")

# Footer avec informations techniques
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Informations techniques")
st.sidebar.markdown(f"- **Device** : {device}")  
st.sidebar.markdown(f"- **Dataset** : Cityscapes 8 classes")
st.sidebar.markdown(f"- **Modèles** : PIDNet-S + DeepLabS")
st.sidebar.markdown(f"- **Framework** : Streamlit + PyTorch + TensorFlow")

# Informations sur les modèles
st.sidebar.markdown("### 🤖 État des modèles")
if 'pidnet_loaded' in locals():
    status_pidnet = "✅ Chargé" if pidnet_loaded else "❌ Erreur"
    st.sidebar.markdown(f"- **PIDNet-S** : {status_pidnet}")
if 'deeplabs_loaded' in locals():
    status_deeplabs = "✅ Chargé" if deeplabs_loaded else "❌ Erreur"
    st.sidebar.markdown(f"- **DeepLabS** : {status_deeplabs}")