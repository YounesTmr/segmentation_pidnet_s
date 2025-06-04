import sys
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import os

# Ajoute le dossier contenant pidnet_s dans le PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), 'pidnet_s'))
from models.pidnet import pidnet_s

def load_pidnet_model(checkpoint_path, device):
    model = pidnet_s(num_classes=8, pretrained=False)  # 8 classes
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Si le checkpoint est un dict complet (ex : checkpoint['model_state_dict'])
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    return model

def predict_segmentation(model, image, device):
    transform = transforms.Compose([
        transforms.Resize((512, 1024)),  # Taille standard Cityscapes
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)[0]  # shape (batch_size, 8, H, W)
    pred = torch.argmax(output.squeeze(), dim=0).cpu().numpy()  # prédiction 8 classes
    return pred
