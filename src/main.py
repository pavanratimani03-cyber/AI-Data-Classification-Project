"""
===============================================================================
AI Project 2: Data Classification Using AI
Dataset: Iris Dataset
Algorithms: K-Nearest Neighbors (KNN - Primary) & Decision Tree (Benchmark)
Author: AI Internship Student
===============================================================================
This script performs a complete machine learning classification workflow:
1. Dataset Loading & Export
2. Data Preprocessing & Inspection
3. Train/Test Split & Feature Scaling
4. Supervised Model Training (KNN & Decision Tree)
5. Model Evaluation (Accuracy, Confusion Matrix, Classification Report)
6. Plot Generation (Saved to results/)
7. Interactive & Sample Predictions on New User Data
===============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def setup_directories(base_dir):
    """Ensure data/ and results/ directories exist."""
    data_dir = os.path.join(base_dir, "data")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    return data_dir, results_dir


def load_and_prepare_dataset(data_dir):
    """
    Load the Iris dataset, save to data/iris.csv for standalone execution,
    and return as a pandas DataFrame.
    """
    csv_path = os.path.join(data_dir, "iris.csv")
    
    if not os.path.exists(csv_path):
        print("[+] Fetching Iris dataset from scikit-learn...")
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['target'] = iris.target
        df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        df.to_csv(csv_path, index=False)
        print(f"[+] Saved dataset locally to: {csv_path}")
    else:
        print(f"[+] Loading dataset from local CSV: {csv_path}")
        df = pd.read_csv(csv_path)

    return df


def inspect_dataset(df):
    """Print key dataset information and statistical metrics."""
    print("\n" + "=" * 60)
    print("STEP 1: DATASET INSPECTION & UNDERSTANDING")
    print("=" * 60)
    
    print(f"Dataset Shape: {df.shape[0]} samples (rows), {df.shape[1]} columns")
    
    print("\n--- First 5 Records (Sample Data) ---")
    print(df.head())
    
    print("\n--- Column Data Types & Non-Null Counts ---")
    print(df.info())
    
    print("\n--- Missing Values Check ---")
    null_counts = df.isnull().sum()
    print(null_counts)
    if null_counts.sum() == 0:
        print("-> Clean Dataset: Zero missing values found!")
        
    print("\n--- Target Class Distribution ---")
    print(df['species'].value_counts())
    
    print("\n--- Feature Summary Statistics ---")
    feature_cols = [col for col in df.columns if col not in ['target', 'species']]
    print(df[feature_cols].describe().T[['mean', 'std', 'min', '50%', 'max']])


def preprocess_and_split(df):
    """
    Select features and target, split into train and test sets (80/20),
    and scale features using StandardScaler.
    """
    print("\n" + "=" * 60)
    print("STEP 2: FEATURE SELECTION, TRAIN/TEST SPLIT & SCALING")
    print("=" * 60)
    
    # Feature columns (independent variables)
    feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    
    X = df[feature_names]
    y = df['target']
    
    print(f"Features Matrix (X) Shape: {X.shape}")
    print(f"Target Vector (y) Shape:   {y.shape}")
    
    # 80% Training set, 20% Testing set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"Training Samples: {X_train.shape[0]} | Testing Samples: {X_test.shape[0]}")
    
    # Standardize features (mean=0, variance=1) for distance calculation in KNN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X, y, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler


def train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test, results_dir):
    """
    Train K-Nearest Neighbors (Primary) and Decision Tree (Benchmark).
    Evaluate both models on training and test accuracy, matrix, and reports.
    """
    print("\n" + "=" * 60)
    print("STEP 3: MODEL TRAINING & EVALUATION")
    print("=" * 60)
    
    species_names = ['setosa', 'versicolor', 'virginica']
    
    # -------------------------------------------------------------
    # 1. Primary Model: K-Nearest Neighbors (KNN, k=5)
    # -------------------------------------------------------------
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_scaled, y_train)
    
    knn_train_pred = knn_model.predict(X_train_scaled)
    knn_test_pred = knn_model.predict(X_test_scaled)
    
    knn_train_acc = accuracy_score(y_train, knn_train_pred)
    knn_test_acc = accuracy_score(y_test, knn_test_pred)
    
    print("\n[MODEL 1: K-NEAREST NEIGHBORS (KNN, k=5)]")
    print(f"--> Training Accuracy: {knn_train_acc * 100:.2f}%")
    print(f"--> Testing Accuracy:  {knn_test_acc * 100:.2f}%")
    
    cm_knn = confusion_matrix(y_test, knn_test_pred)
    print("\nConfusion Matrix (KNN):")
    print(pd.DataFrame(cm_knn, index=[f"Actual {s}" for s in species_names], columns=[f"Pred {s}" for s in species_names]))
    
    print("\nClassification Report (KNN):")
    print(classification_report(y_test, knn_test_pred, target_names=species_names))
    
    # -------------------------------------------------------------
    # 2. Benchmark Model: Decision Tree Classifier
    # -------------------------------------------------------------
    dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_model.fit(X_train_scaled, y_train)
    
    dt_train_pred = dt_model.predict(X_train_scaled)
    dt_test_pred = dt_model.predict(X_test_scaled)
    
    dt_train_acc = accuracy_score(y_train, dt_train_pred)
    dt_test_acc = accuracy_score(y_test, dt_test_pred)
    
    print("\n[MODEL 2: DECISION TREE CLASSIFIER (max_depth=3)]")
    print(f"--> Training Accuracy: {dt_train_acc * 100:.2f}%")
    print(f"--> Testing Accuracy:  {dt_test_acc * 100:.2f}%")
    
    # -------------------------------------------------------------
    # Visualizations (Saved to results/ directory)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: GENERATING & SAVING VISUALIZATIONS")
    print("=" * 60)
    
    # Plot 1: Confusion Matrix Heatmap
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues',
                xticklabels=species_names, yticklabels=species_names)
    plt.title('KNN Classification - Confusion Matrix', fontsize=14, pad=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[+] Confusion matrix plot saved to: {cm_path}")
    
    return knn_model, dt_model, knn_test_pred


def test_new_samples(knn_model, scaler, feature_names):
    """
    Test the trained KNN model using new custom sample measurements.
    """
    print("\n" + "=" * 60)
    print("STEP 5: PREDICTING WITH NEW / UNSEEN INPUT DATA")
    print("=" * 60)
    
    species_map = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
    
    # Sample Test Cases representing distinct species profiles
    sample_inputs = [
        {"name": "Sample 1 (Typical Setosa)",     "data": [5.1, 3.5, 1.4, 0.2]},
        {"name": "Sample 2 (Typical Versicolor)", "data": [5.7, 2.8, 4.1, 1.3]},
        {"name": "Sample 3 (Typical Virginica)",  "data": [6.9, 3.1, 5.4, 2.1]}
    ]
    
    print("\n--- Running Predictions on Pre-defined Test Samples ---")
    for sample in sample_inputs:
        input_df = pd.DataFrame([sample["data"]], columns=feature_names)
        input_scaled = scaler.transform(input_df)
        
        prediction = knn_model.predict(input_scaled)[0]
        probabilities = knn_model.predict_proba(input_scaled)[0]
        predicted_species = species_map[prediction]
        confidence = probabilities[prediction] * 100
        
        print(f"\n{sample['name']}:")
        print(f"  Measurements (cm): Sepal Len={sample['data'][0]}, Sepal Wid={sample['data'][1]}, Petal Len={sample['data'][2]}, Petal Wid={sample['data'][3]}")
        print(f"  --> Predicted Class: {predicted_species}")
        print(f"  --> Prediction Confidence: {confidence:.2f}%")


def interactive_prediction(knn_model, scaler, feature_names):
    """Allows user to manually enter measurements via keyboard input."""
    print("\n" + "-" * 60)
    print("OPTIONAL INTERACTIVE PREDICTION MODE")
    print("-" * 60)
    species_map = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
    
    try:
        user_choice = input("Would you like to enter custom flower measurements? (y/n): ").strip().lower()
        if user_choice == 'y':
            print("\nEnter measurements in centimeters:")
            sl = float(input("Sepal Length (cm) [e.g. 5.1]: "))
            sw = float(input("Sepal Width (cm)  [e.g. 3.5]: "))
            pl = float(input("Petal Length (cm) [e.g. 1.4]: "))
            pw = float(input("Petal Width (cm)  [e.g. 0.2]: "))
            
            user_df = pd.DataFrame([[sl, sw, pl, pw]], columns=feature_names)
            user_scaled = scaler.transform(user_df)
            
            pred = knn_model.predict(user_scaled)[0]
            probs = knn_model.predict_proba(user_scaled)[0]
            
            print(f"\n[RESULT] Predicted Species: {species_map[pred]}")
            print(f"[CONFIDENCE] {probs[pred] * 100:.2f}%")
        else:
            print("Skipping interactive mode.")
    except Exception as e:
        print(f"Invalid input or operation cancelled: {e}")


def main():
    print("\n" + "=" * 70)
    print("  AI PROJECT 2: DATA CLASSIFICATION USING K-NEAREST NEIGHBORS (KNN)")
    print("=" * 70)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir, results_dir = setup_directories(base_dir)
    
    # 1. Load dataset
    df = load_and_prepare_dataset(data_dir)
    
    # 2. Inspect dataset
    inspect_dataset(df)
    
    # 3. Preprocess & Train/Test Split
    X, y, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler = preprocess_and_split(df)
    
    # 4. Train & Evaluate
    knn_model, dt_model, test_preds = train_and_evaluate_models(
        X_train_scaled, X_test_scaled, y_train, y_test, results_dir
    )
    
    # 5. Predict on New Data
    test_new_samples(knn_model, scaler, feature_names)
    
    # 6. Interactive option (non-blocking if redirected)
    if sys.stdin.isatty():
        interactive_prediction(knn_model, scaler, feature_names)
        
    print("\n" + "=" * 70)
    print("  SUCCESS: AI DATA CLASSIFICATION PIPELINE EXECUTED CLEANLY!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
