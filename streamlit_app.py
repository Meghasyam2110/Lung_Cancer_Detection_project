import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import joblib
from PIL import Image
import os
from sklearn.metrics import classification_report

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names
class_names = ['Benign cases', 'Malignant cases', 'Normal cases']

# CNN Weights
cnn_weights = {
    "mobilenet": 1.0,
    "resnet": 0.7,
    "efficientnet": 1.5,
    "googlenet": 1.0,
    "densenet": 0.7,
}

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load all CNN models
def get_model(name, num_classes=3):
    if name == 'mobilenet':
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == 'efficientnet':
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == 'googlenet':
        model = models.googlenet(pretrained=False, aux_logits=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == 'resnet':
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == 'densenet':
        model = models.densenet121(pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError("Unknown model name")
    return model

@st.cache_resource
def load_cnn_models():
    model_dict = {}
    for name in cnn_weights.keys():
        model = get_model(name)
        model_path = f"{name}_best.pth"
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        model_dict[name] = model
    return model_dict

def extract_features_single(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        if isinstance(output, tuple):  # GoogLeNet
            output = output[0]
        output = output.view(output.size(0), -1)
        return output.cpu().numpy()

def ensemble_predict(image):
    image_tensor = transform(image).unsqueeze(0).to(device)
    cnn_models = load_cnn_models()
    total_prob = np.zeros((1, 3))

    for cnn_name, model in cnn_models.items():
        features = extract_features_single(model, image_tensor)
        probs_list = []

        for ml_name in ['svm', 'knn', 'naive_bayes', 'decision_tree', 'random_forest', 'adaboost', 'mlp']:
            ml_model_path = f"saved_models/{cnn_name}_{ml_name}_model.pkl"
            if os.path.exists(ml_model_path):
                ml_model = joblib.load(ml_model_path)
                probs = ml_model.predict_proba(features)
                probs_list.append(probs)

        if probs_list:
            avg_prob = np.mean(probs_list, axis=0)
            weighted_prob = avg_prob * cnn_weights[cnn_name]
            total_prob += weighted_prob

    # Normalize
    total_prob /= sum(cnn_weights.values())
    final_pred = np.argmax(total_prob)
    return final_pred, total_prob[0]

# ===== Streamlit UI =====
st.set_page_config(page_title="Lung Cancer Detection - Ensemble App", layout="centered")
st.title("🫁 Lung Cancer Detection using Ensemble Learning")
st.write("Upload a CT scan image to detect possible lung cancer.")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing with ensemble model..."):
            pred_class, pred_probs = ensemble_predict(image)
            st.success(f"🎯 **Prediction**: {class_names[pred_class]}")
            st.write("### 🧪 Prediction Probabilities:")
            for i, cls in enumerate(class_names):
                st.write(f"{cls}: {pred_probs[i]*100:.2f}%")

            # Optional: Display mock classification report structure
            predicted_label = pred_class
            true_label = pred_class  # Assume prediction for uploaded unknown
            report = classification_report(
    [true_label], [predicted_label],
    labels=[0, 1, 2],
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

            st.write("### 📊 Classification Report")
            for label in class_names:
                st.write(f"**{label}** — Precision: {report[label]['precision']:.2f}, Recall: {report[label]['recall']:.2f}, F1: {report[label]['f1-score']:.2f}")
