# Diabetes Prediction System - ML Final Exam

A comprehensive machine learning project that predicts diabetes risk using patient medical data. This project implements data preprocessing, model selection, hyperparameter tuning, and a user-friendly web interface.

## 📊 Project Overview

This final exam project demonstrates machine learning best practices including:
- **Data Preprocessing**: Handling missing values and zero values properly
- **Model Comparison**: Evaluating 6 different ML algorithms
- **Hyperparameter Tuning**: GridSearchCV optimization for all models
- **Cross-Validation**: 5-fold CV for robust performance estimation
- **Web Interface**: Interactive Gradio-based prediction system

## 📁 Project Structure

```
├── final_exam.py              # Main ML pipeline and training script
├── app.py                     # Gradio web interface for predictions
├── diabetes.csv               # Dataset (768 samples, 9 features)
├── final_exam_model.pkl       # Trained model (generated after running final_exam.py)
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 📋 Dataset Details

**Source**: Pima Indians Diabetes Database

**Features (8)**:
- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration (2 hours post-meal)
- **BloodPressure**: Diastolic blood pressure (mmHg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg / height in m²)
- **DiabetesPedigreeFunction**: Diabetes pedigree function score
- **Age**: Age (years)

**Target**: Outcome (0 = Non-Diabetic, 1 = Diabetic)

**Size**: 768 samples

## 🔧 Technical Implementation

### 1. Data Preprocessing (10 marks)
- ✅ Check for missing values
- ✅ Replace zero values with training data median (prevents data leakage)
- ✅ Feature/Target split with stratification
- ✅ Train-test split: 80/20 ratio with stratification

### 2. Pipeline Creation (10 marks)
- StandardScaler for feature normalization
- Pipeline structure for reproducibility
- Consistent preprocessing across train and test sets

### 3. Model Selection & Training (15 marks)
Six models evaluated with 5-fold cross-validation:

| Model | Best For |
|-------|----------|
| **Logistic Regression** | Baseline, interpretability |
| **Decision Tree** | Non-linear patterns |
| **Random Forest** | Ensemble, feature importance |
| **Gradient Boosting** | Complex patterns, high accuracy |
| **SVM** | Non-linear boundaries |
| **KNN** | Local pattern detection |

### 4. Hyperparameter Tuning (10 marks)
GridSearchCV applied to all 6 models:
- **Logistic Regression**: C, max_iter
- **Decision Tree**: max_depth, min_samples_split
- **Random Forest**: n_estimators, max_depth
- **Gradient Boosting**: n_estimators, learning_rate
- **SVM**: C, kernel
- **KNN**: n_neighbors, weights

### 5. Model Evaluation (10 marks)
- Accuracy Score
- F1 Score
- Confusion Matrix
- Classification Report
- Cross-validation metrics

### 6. Web Interface (10 marks)
Interactive Gradio interface for real-time predictions

### 7. Model Persistence (5 marks)
Trained model saved as pickle file for deployment

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone/Download the project**
```bash
cd "c:\JISAN\Python\AI_ML Course\Final Exam"
```

2. **Create virtual environment (recommended)**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 📖 Usage

### Step 1: Train the Model
```bash
python final_exam.py
```

This script will:
1. Load and preprocess the diabetes dataset
2. Perform 5-fold cross-validation on all 6 models
3. Display model comparison results
4. Run hyperparameter tuning on the best model
5. Evaluate performance on test set
6. Launch interactive Gradio web interface
7. Save the trained model as `final_exam_model.pkl`

### Step 2: Use the Web Interface
The Gradio interface will launch automatically and display:
- Input fields for 8 patient features
- Real-time prediction: "Diabetic" or "Non-Diabetic"
- Shareable link for remote access

### Step 3: Use Pre-trained Model (app.py)
```bash
python app.py
```
This loads the saved model and launches the Gradio interface without retraining.

## 📊 Expected Results

### Cross-Validation Metrics
The model achieves strong performance through:
- Multiple algorithm comparison
- Balanced class weighting (handles class imbalance)
- Stratified sampling
- Comprehensive hyperparameter optimization

### Model Output Example
```
Input: Pregnancies=6, Glucose=148, BloodPressure=72, SkinThickness=35, 
       Insulin=0, BMI=33.6, DiabetesPedigreeFunction=0.627, Age=50
Output: Diabetic (1)
```

## 🔍 Key Features

✅ **Data Leakage Prevention**: Zero-value replacement happens after train-test split
✅ **Class Balancing**: `class_weight="balanced"` in classifiers
✅ **Stratified Sampling**: Maintains class distribution in train/test split
✅ **Comprehensive Tuning**: All 6 models get hyperparameter optimization
✅ **Pipeline Integration**: Consistent preprocessing via scikit-learn Pipeline
✅ **Cross-Validation**: 5-fold CV for robust metric estimation
✅ **Model Persistence**: Pickle serialization for deployment

## 📦 Dependencies

Key packages:
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **scikit-learn**: ML algorithms and utilities
- **gradio**: Web interface
- **pickle**: Model serialization

See `requirements.txt` for complete list and versions.

## 🎯 Project Marks Breakdown (100 marks total)

- Data Loading: 5 marks
- Data Preprocessing: 10 marks
- Pipeline Creation: 10 marks
- Primary Model Selection: 5 marks
- Model Training: 10 marks
- Cross-Validation: 10 marks
- Hyperparameter Tuning: 10 marks
- Best Model Selection: 10 marks
- Model Performance Evaluation: 10 marks
- Web Interface with Gradio: 10 marks

## 💡 Important Notes

### Data Preprocessing Strategy
- Zero values in medical features are replaced with training data median
- This prevents data leakage since test data doesn't influence preprocessing
- Outliers (high Insulin/Glucose) are retained as valid medical measurements

### Model Selection Logic
- Models are compared using mean accuracy and F1 score
- Best model is selected after cross-validation
- Selected model undergoes hyperparameter tuning
- Final evaluation is performed on held-out test set

### Why This Approach?
1. **Prevents overfitting** through cross-validation
2. **Prevents data leakage** through proper data handling
3. **Fair comparison** by tuning all models equally
4. **Production ready** with model persistence and web interface

## 🐛 Troubleshooting

**Issue**: `FileNotFoundError: diabetes.csv not found`
- **Solution**: Ensure `diabetes.csv` is in the same directory as `final_exam.py`

**Issue**: Gradio interface won't launch
- **Solution**: Check port 7860 is available, or Gradio will use next available port

**Issue**: Model prediction errors
- **Solution**: Ensure all input values are numerical and in valid ranges

## 📝 License

This is an academic project for AI/ML course final examination.

## 👨‍💻 Author

AI/ML Course - Final Exam Project

---

**Last Updated**: January 2026
