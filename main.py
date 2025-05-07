#Data Augmentation

import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

# Set root path
dataset_root = r"D:\IQ-Processed-NoDuplicates"

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

])

# Load datasets
train_dataset = ImageFolder(root=os.path.join(dataset_root, "train"), transform=train_transform)
val_dataset = ImageFolder(root=os.path.join(dataset_root, "val"), transform=val_test_transform)
test_dataset = ImageFolder(root=os.path.join(dataset_root, "test"), transform=val_test_transform)

# Class balancing using WeightedRandomSampler
targets = train_dataset.targets
class_counts = np.bincount(targets)
class_weights = 1. / class_counts
sample_weights = [class_weights[label] for label in targets]

sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Print dataset info
print("Classes:", train_dataset.classes)
print("Train:", len(train_dataset), "| Val:", len(val_dataset), "| Test:", len(test_dataset))
print("Class counts in train:", dict(zip(train_dataset.classes, class_counts)))



#CNN Training

import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

# Set root path
dataset_root = r"D:\IQ-Processed-NoDuplicates"

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

])

# Load datasets
train_dataset = ImageFolder(root=os.path.join(dataset_root, "train"), transform=train_transform)
val_dataset = ImageFolder(root=os.path.join(dataset_root, "val"), transform=val_test_transform)
test_dataset = ImageFolder(root=os.path.join(dataset_root, "test"), transform=val_test_transform)

# Class balancing using WeightedRandomSampler
targets = train_dataset.targets
class_counts = np.bincount(targets)
class_weights = 1. / class_counts
sample_weights = [class_weights[label] for label in targets]

sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Print dataset info
print("Classes:", train_dataset.classes)
print("Train:", len(train_dataset), "| Val:", len(val_dataset), "| Test:", len(test_dataset))
print("Class counts in train:", dict(zip(train_dataset.classes, class_counts)))

#loading trained CNN

import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

# Set root path
dataset_root = r"D:\IQ-Processed-NoDuplicates"

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

])

# Load datasets
train_dataset = ImageFolder(root=os.path.join(dataset_root, "train"), transform=train_transform)
val_dataset = ImageFolder(root=os.path.join(dataset_root, "val"), transform=val_test_transform)
test_dataset = ImageFolder(root=os.path.join(dataset_root, "test"), transform=val_test_transform)

# Class balancing using WeightedRandomSampler
targets = train_dataset.targets
class_counts = np.bincount(targets)
class_weights = 1. / class_counts
sample_weights = [class_weights[label] for label in targets]

sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Print dataset info
print("Classes:", train_dataset.classes)
print("Train:", len(train_dataset), "| Val:", len(val_dataset), "| Test:", len(test_dataset))
print("Class counts in train:", dict(zip(train_dataset.classes, class_counts)))


# Feature extraction and ml models

import torch
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import os

# Set your device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create directory for saved models if it doesn't exist
os.makedirs("saved_models", exist_ok=True)

# ML classifiers
ml_models = {
    "svm": SVC(kernel="linear", probability=True),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "naive_bayes": GaussianNB(),
    "decision_tree": DecisionTreeClassifier(),
    "random_forest": RandomForestClassifier(n_estimators=100),
    "adaboost": AdaBoostClassifier(n_estimators=50),
    "mlp": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
}

# Custom CNN weights
cnn_weights = {
    "mobilenet": 1.0,
    "resnet": 0.7,
    "efficientnet": 1.5,
    "googlenet": 1.0,
    "densenet": 0.7,
}

# Feature extraction function
def extract_features(model, dataloader):
    model.eval()
    features, labels = [], []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            if isinstance(outputs, tuple):  # for models like GoogLeNet
                outputs = outputs[0]

            outputs = outputs.view(outputs.size(0), -1)  # Flatten features
            features.append(outputs.cpu().numpy())
            labels.append(targets.numpy())

    features = np.vstack(features)
    labels = np.hstack(labels)
    return features, labels

# Dictionary to store prediction probabilities
final_probabilities = {}

# Train ML models on CNN features and save them
for cnn_name, cnn_model in trained_models.items():
    print(f"\n--- Extracting features from {cnn_name.upper()} ---")
    cnn_model.to(device)
    X_features, y_labels = extract_features(cnn_model, val_loader)

    print(f"--- Training and saving ML models on {cnn_name.upper()} features ---")
    model_probabilities = []

    for ml_name, ml_model in ml_models.items():
        # Train the model
        ml_model.fit(X_features, y_labels)
        
        # Save the trained ML model
        ml_model_path = f"saved_models/{cnn_name}_{ml_name}_model.pkl"
        joblib.dump(ml_model, ml_model_path)
        print(f"Saved {ml_name} model for {cnn_name} at {ml_model_path}")
        
        # Get predictions
        y_prob = ml_model.predict_proba(X_features)
        acc = accuracy_score(y_labels, np.argmax(y_prob, axis=1))
        print(f"{ml_name} on {cnn_name}: {acc:.4f}")
        model_probabilities.append(y_prob)

    final_probabilities[cnn_name] = np.array(model_probabilities)

    # Save the features for later use if needed
    np.savez(f"saved_models/{cnn_name}_features.npz", 
             features=X_features, 
             labels=y_labels)

# Save the CNN models if they're not already saved
for cnn_name, cnn_model in trained_models.items():
    model_path = f"saved_models/{cnn_name}_model.pth"
    if not os.path.exists(model_path):
        torch.save(cnn_model.state_dict(), model_path)
        print(f"Saved {cnn_name} model at {model_path}")

# Ensemble Voting
voted_probabilities = np.zeros_like(final_probabilities["googlenet"][0])

for cnn_name, probs in final_probabilities.items():
    weight = cnn_weights[cnn_name]
    avg_prob = np.mean(probs, axis=0) * weight
    voted_probabilities += avg_prob

# Normalize and predict final class
voted_probabilities /= sum(cnn_weights.values())
voted_predictions = np.argmax(voted_probabilities, axis=1)

# Final ensemble accuracy
ensemble_accuracy = accuracy_score(y_labels, voted_predictions)
print(f"\n✅ Final Weighted Ensemble Accuracy (Validation Set): {ensemble_accuracy:.4f}")

# Save the ensemble voting weights for deployment
ensemble_weights = {
    'cnn_weights': cnn_weights,
    'class_names': ['Benign cases', 'Malignant cases', 'Normal cases']
}
joblib.dump(ensemble_weights, 'saved_models/ensemble_weights.pkl')
print("Saved ensemble weights configuration")
