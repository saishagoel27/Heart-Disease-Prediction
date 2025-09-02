# Heart Disease Prediction using Deep Learning ❤️

## Project Overview
This project demonstrates the application of deep learning techniques for medical diagnosis, specifically predicting heart disease using patient clinical data. The implementation showcases a complete machine learning pipeline from data preprocessing to neural network deployment, making it an excellent learning resource for healthcare AI applications.

## Dataset Features
The dataset contains 13 clinical features commonly used in cardiac assessments:

### Numerical Features
- **age**: Patient's age in years
- **trestbps**: Resting blood pressure (mm Hg)
- **chol**: Serum cholesterol level (mg/dl)
- **thalach**: Maximum heart rate achieved
- **oldpeak**: ST depression induced by exercise

### Categorical Features
- **sex**: Gender (1 = male, 0 = female)
- **cp**: Chest pain type (0-3)
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- **restecg**: Resting electrocardiographic results (0-2)
- **exang**: Exercise induced angina (1 = yes, 0 = no)
- **slope**: Slope of peak exercise ST segment (0-2)
- **ca**: Number of major vessels colored by fluoroscopy (0-3)
- **thal**: Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)

**Target**: Binary classification (1 = heart disease present, 0 = no heart disease)

## Implementation Workflow

### 1. Environment Setup
```python
# Install required libraries
!pip install tensorflow scikit-learn pandas matplotlib seaborn streamlit

# Import essential libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
```

### 2. Data Exploration and Analysis
```python
# Load and examine the dataset
df = pd.read_csv("heart.csv")

# Comprehensive data exploration
df.info()      # Data types and null values
df.describe()  # Statistical summaries
df.head()      # First few records
df.tail()      # Last few records
```

**Learning Point**: Always start with thorough data exploration to understand distributions, potential outliers, and data quality issues.

### 3. Visualization and Pattern Recognition
```python
# Correlation analysis
plt.subplot(2, 2, 4)
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, cbar=False)
plt.title("Correlation Heatmap")

# Feature-target relationships
sns.boxplot(x='target', y='thalach', data=df, palette='viridis')
plt.title("Maximum Heart Rate Distribution")

sns.boxplot(x='target', y='chol', data=df, palette='viridis')
plt.title("Cholesterol Levels")
```

**Learning Concept**: Visualization helps identify which features show clear separation between classes, guiding feature selection decisions.

### 4. Advanced Data Preprocessing
```python
# Separate features and target
X = df.drop(columns=['target'])
y = df['target']

# Define feature types for preprocessing
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Create preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(), categorical_features)
])

# Transform the data
X_processed = preprocessor.fit_transform(X)
```

**Key Learning**: The ColumnTransformer allows different preprocessing for different feature types, which is crucial for mixed data types in medical datasets.

### 5. Stratified Data Splitting
```python
# Split with stratification to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)
```

**Learning Point**: Stratified splitting ensures both training and test sets have similar class distributions, critical for medical applications.

### 6. Multi-Layer Perceptron Architecture
```python
# Build neural network architecture
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),    # Hidden layer 1
    layers.Dense(32, activation='relu'),    # Hidden layer 2
    layers.Dense(1, activation='sigmoid')   # Output layer
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

**Architecture Explanation**:
- **Input Layer**: Accepts all processed features
- **Hidden Layers**: 64 and 32 neurons with ReLU activation for non-linearity
- **Output Layer**: Single neuron with sigmoid activation for binary classification

### 7. Model Training and Validation
```python
# Train with validation monitoring
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32
)

# Evaluate final performance
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Save the trained model
model.save("heart_disease_model.h5")
```

## Key Learning Concepts

### 1. **Medical Data Preprocessing**
- Handling mixed data types (numerical and categorical)
- Standardization for neural networks
- One-hot encoding for categorical variables
- Stratified sampling for imbalanced medical data

### 2. **Deep Learning for Healthcare**
- Binary classification for diagnostic applications
- Sigmoid activation for probability outputs
- Adam optimizer for medical data
- Cross-validation during training

### 3. **Neural Network Architecture Design**
- Input layer dimensionality based on processed features
- Hidden layer sizing (decreasing neuron count)
- Activation functions for medical classification
- Model compilation strategies

### 4. **Model Evaluation in Healthcare**
- Accuracy metrics for diagnostic tools
- Validation set monitoring during training
- Model persistence for deployment

## Clinical Significance
This model demonstrates how machine learning can assist in:
- **Early Detection**: Identifying high-risk patients
- **Resource Allocation**: Prioritizing patients for further testing
- **Decision Support**: Providing additional information to healthcare providers
- **Screening Programs**: Population-level health assessments

## Extensions for Advanced Learning
- **Performance Metrics**: Add precision, recall, F1-score, AUC-ROC
- **Cross-Validation**: Implement k-fold cross-validation
- **Hyperparameter Tuning**: Grid search or Bayesian optimization
- **Feature Importance**: Analyze which clinical features matter most
- **Model Comparison**: Compare with Random Forest, SVM, Logistic Regression
- **Deployment**: Create a Streamlit web application for predictions

## Requirements
```
tensorflow
scikit-learn
pandas
matplotlib
seaborn
streamlit
```

## Real-World Application
This project simulates the development process for clinical decision support systems, demonstrating:
- Data preprocessing pipelines for medical records
- Neural network design for diagnostic applications
- Model validation strategies for healthcare AI
- Deployment considerations for medical software

---

*This project serves as an educational foundation for understanding how deep learning can be applied to healthcare challenges while emphasizing the importance of rigorous validation and ethical considerations in medical AI*
