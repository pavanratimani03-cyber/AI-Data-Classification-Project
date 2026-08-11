# Artificial Intelligence Project 2: Data Classification Using AI

## 1. Project Title
**Data Classification Using AI: Iris Flower Species Classification**

---

## 2. Project Objective
The primary objective of this project is to construct an end-to-end, beginner-friendly **Supervised Machine Learning Classification Pipeline**. The system loads a structured tabular dataset, inspects and preprocesses the data, splits it into training and testing sets, trains a **K-Nearest Neighbors (KNN)** classification model, evaluates model performance using standard metrics (Accuracy, Confusion Matrix, Classification Report), and makes predictions on new/unseen user inputs.

---

## 3. Problem Statement
Given physical flower measurements (sepal length, sepal width, petal length, petal width), automatically and accurately classify an Iris flower into one of three species:
1. **Iris Setosa**
2. **Iris Versicolor**
3. **Iris Virginica**

Automating flower species identification using machine learning avoids human error, speeds up biological classification, and serves as a classic foundational paradigm for tabular data classification in Artificial Intelligence.

---

## 4. Technologies Used
- **Python (v3.10+)**: Core programming language.
- **Pandas**: Data manipulation, DataFrame structure, CSV export/import, and summary statistics.
- **NumPy**: Numerical computations and array indexing.
- **Scikit-learn (sklearn)**: Machine learning library for data splitting (`train_test_split`), feature scaling (`StandardScaler`), model algorithms (`KNeighborsClassifier`, `DecisionTreeClassifier`), and performance metrics.
- **Matplotlib & Seaborn**: Data visualization and saving confusion matrix heatmaps to the `results/` directory.

---

## 5. Dataset Description
The project uses the classic **Iris Dataset** created by Ronald Fisher (1936):
- **Total Samples**: 150 instances (50 samples per class).
- **Attributes / Features (4 Independent Variables)**:
  1. `sepal length (cm)`
  2. `sepal width (cm)`
  3. `petal length (cm)`
  4. `petal width (cm)`
- **Target Label (1 Dependent Variable)**:
  - `0`: Setosa
  - `1`: Versicolor
  - `2`: Virginica
- **Missing Values**: 0 (Clean, complete dataset ideal for educational demonstration).

---

## 6. Machine-Learning Algorithm Used
- **Primary Algorithm**: **K-Nearest Neighbors (KNN)** with `n_neighbors = 5`.
- **Reasoning**:
  - **Intuitive & Visual**: KNN operates on a distance metric (Euclidean distance). It classifies a new point based on the majority vote of its 5 closest neighbors in feature space.
  - **No Complex Mathematical Assumptions**: Ideal for internship students to explain during project defense/viva.
  - **High Performance**: Produces near 100% accuracy on linearly & non-linearly separable flower morphometrics.
- **Benchmark Algorithm**: **Decision Tree Classifier** (`max_depth=3`), included to showcase algorithm benchmarking fundamentals.

---

## 7. Step-by-Step Methodology

```
┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│ 1. Load Data (CSV/Sklearn)│ ──> │ 2. Inspect & Preprocess │ ──> │ 3. Train / Test Split   │
└─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
                                                                             │
┌─────────────────────────┐     ┌─────────────────────────┐                  ▼
│ 6. Predict New Samples  │ <── │ 5. Evaluate Metrics     │ <── ┌─────────────────────────┐
│    & Visual Matrix Plot │     │    (Accuracy/Confusion) │     │ 4. Train KNN Model      │
└─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
```

1. **Dataset Loading**: Export/load dataset into a Pandas DataFrame (`data/iris.csv`).
2. **Inspection**: Verify shape (`150x5`), inspect `.head()`, check null values (`.isnull().sum()`), and calculate feature distributions (`.describe()`).
3. **Train-Test Split**: Divide dataset into 80% Training Set (120 samples) and 20% Testing Set (30 samples) with stratified sampling.
4. **Feature Scaling**: Standardize feature ranges using `StandardScaler` ($\mu=0, \sigma=1$) to prevent features with larger scales from dominating KNN distance calculations.
5. **Model Training**: Fit `KNeighborsClassifier` on `X_train_scaled` and `y_train`.
6. **Model Evaluation**: Test model on unseen `X_test_scaled` data, compute training & testing accuracy, print confusion matrix, and generate classification reports.
7. **Visualization**: Render and save the confusion matrix heatmap image to `results/confusion_matrix.png`.
8. **Inference / Prediction**: Feed custom sample measurements to output predicted species and class probabilities.

---

## 8. Project Structure

```
AI_Data_Classification/
│
├── data/
│   └── iris.csv               # Standalone dataset file
├── results/
│   └── confusion_matrix.png   # Saved evaluation heatmap plot
├── src/
│   └── main.py                # Main executable pipeline script
├── README.md                  # Complete project documentation & viva guide
└── requirements.txt           # Python dependencies
```

---

## 9. Installation Instructions

1. Open your terminal / command prompt.
2. Navigate to the project root directory:
   ```bash
   cd AI_Data_Classification
   ```
3. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## 10. How to Run the Project

Run the main execution script:
```bash
python src/main.py
```

The script will automatically perform data loading, preprocessing, model training, evaluation, plot generation into `results/`, and display sample predictions.

---

## 11. Sample Output

```text
======================================================================
  AI PROJECT 2: DATA CLASSIFICATION USING K-NEAREST NEIGHBORS (KNN)
======================================================================
[+] Saved dataset locally to: data/iris.csv

============================================================
STEP 1: DATASET INSPECTION & UNDERSTANDING
============================================================
Dataset Shape: 150 samples (rows), 6 columns

--- First 5 Records (Sample Data) ---
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target species
0                5.1               3.5                1.4               0.2       0  setosa
1                4.9               3.0                1.4               0.2       0  setosa
2                4.7               3.2                1.3               0.2       0  setosa
...

--- Missing Values Check ---
Clean Dataset: Zero missing values found!

============================================================
STEP 3: MODEL TRAINING & EVALUATION
============================================================

[MODEL 1: K-NEAREST NEIGHBORS (KNN, k=5)]
--> Training Accuracy: 97.50%
--> Testing Accuracy:  96.67%

Confusion Matrix (KNN):
                  Pred setosa  Pred versicolor  Pred virginica
Actual setosa              10                0               0
Actual versicolor           0                9               1
Actual virginica            0                0              10

Classification Report (KNN):
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      0.90      0.95        10
   virginica       0.91      1.00      0.95        10

    accuracy                           0.97        30
   macro avg       0.97      0.97      0.97        30
weighted avg       0.97      0.97      0.97        30

============================================================
STEP 5: PREDICTING WITH NEW / UNSEEN INPUT DATA
============================================================

Sample 1 (Typical Setosa):
  Measurements (cm): Sepal Len=5.1, Sepal Wid=3.5, Petal Len=1.4, Petal Wid=0.2
  --> Predicted Class: Iris-setosa
  --> Prediction Confidence: 100.00%
```

---

## 12. Results
- **KNN Model Testing Accuracy**: **96.67%** (29 out of 30 test samples correctly classified).
- **Decision Tree Testing Accuracy**: **96.67%**.
- **Confusion Matrix Analysis**: All Setosa and Virginica samples were identified with 100% precision. High F1-scores across all three flower categories confirm a robust model free from overfitting.

---

## 13. Conclusion
The AI Data Classification Pipeline successfully demonstrates how supervised machine learning algorithms (specifically K-Nearest Neighbors) can automatically classify physical records with high precision. By scaling feature dimensions and utilizing train/test splits, the model generalizes exceptionally well to new unseen flower data.

---

## 14. Future Improvements
1. **Hyperparameter Tuning**: Test different values of $k$ (e.g., $k \in [1, 15]$) using Grid Search to find optimal distance thresholds.
2. **Web Interface / GUI**: Build a web application using Streamlit or Flask so non-technical users can enter measurements via sliders.
3. **Cross-Validation**: Incorporate $k$-fold cross-validation (5-fold or 10-fold) for even stronger statistical validation across smaller datasets.

---

# Internship Project Viva & Presentation Preparation Guide

Use this section to prepare for questions asked during your internship review, viva, or project presentation.

### Q1: What is Supervised Learning?
> **Answer**: Supervised learning is a branch of machine learning where the algorithm is trained using a **labeled dataset**. The training data contains both input features (X) and correct output targets (y). The model learns the underlying mapping function $f(X) = y$ so it can accurately predict outputs for new, unseen input data.

### Q2: What is Classification?
> **Answer**: Classification is a type of supervised learning where the target variable is **categorical** (discrete classes). Instead of predicting a continuous numerical value (which is Regression), classification assigns inputs into distinct categories (e.g., predicting flower species: *Setosa*, *Versicolor*, or *Virginica*).

### Q3: Why was the Iris Dataset selected?
> **Answer**: The Iris dataset is the benchmark standard dataset for beginner AI/ML classification. It has clean data with no missing values, a balanced class distribution (50 samples per species), clear linear and non-linear feature boundaries, and 4 physical features that are easy to comprehend intuitively.

### Q4: Why was K-Nearest Neighbors (KNN) selected?
> **Answer**: KNN is non-parametric, intuitive, and distance-based. It calculates the Euclidean distance between a new input point and all existing training data points, identifies the $k$ closest neighbors (we used $k=5$), and assigns the class by majority vote. It requires no complex underlying probabilistic assumptions and is very easy to explain.

### Q5: What are Features and Labels?
> **Answer**:
> - **Features (X)**: The independent input variables used to make predictions. In our project, these are *sepal length, sepal width, petal length, and petal width*.
> - **Labels (y)**: The dependent target variable/category we want to predict. In our project, this is the flower *species (0: Setosa, 1: Versicolor, 2: Virginica)*.

### Q6: What is Training Data vs. Testing Data?
> **Answer**:
> - **Training Data (80%)**: Data used by the algorithm to learn patterns and build model parameters.
> - **Testing Data (20%)**: Held-out data that the model has never seen during training. It is used to evaluate how well the trained model generalizes to new real-world data.

### Q7: What is Model Accuracy?
> **Answer**: Accuracy is the ratio of correct predictions to the total number of predictions made:
> $$\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Predictions}}$$
> In our project, KNN achieved **96.67% accuracy** on the test set.

### Q8: What is a Confusion Matrix?
> **Answer**: A confusion matrix is a $N \times N$ table used to evaluate the performance of a classification model. The rows represent actual ground-truth classes while the columns represent model predictions. It reveals not only how many predictions were correct, but specifically which classes are being confused with one another.

### Q9: How does the model predict new data?
> **Answer**:
> 1. The user provides 4 measurements $[SL, SW, PL, PW]$.
> 2. The input values are transformed using the fitted `StandardScaler` (so they match the scale of the training data).
> 3. The scaled values are passed to `knn_model.predict()`.
> 4. The model computes distance to all training instances, finds the 5 closest neighbors, and returns the majority class along with confidence probabilities.
