import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# Load Dataset
# -------------------------------
def load_dataset(path="Crop_recommendation.csv"):
    df = pd.read_csv(path)
    return df

# -------------------------------
# Train Models
# -------------------------------
def train_models(X_train, y_train):
    models = {
        "Decision Tree": DecisionTreeClassifier(criterion='entropy', random_state=2, max_depth=5),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(gamma='auto'),
        "Logistic Regression": LogisticRegression(random_state=2),
        "Random Forest": RandomForestClassifier(n_estimators=20, random_state=0)
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        with open(f"{name.replace(' ', '')}.pkl", "wb") as f:
            pickle.dump(model, f)
    return models

# -------------------------------
# Evaluate Model
# -------------------------------
def evaluate_model(model, X_test, y_test):
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    report = classification_report(y_test, pred)
    return acc, report

# -------------------------------
# Visualize Accuracy Comparison
# -------------------------------
def plot_accuracy_comparison(acc_dict):
    models = list(acc_dict.keys())
    scores = list(acc_dict.values())
    plt.figure(figsize=(10, 5))
    plt.title("Accuracy Comparison")
    plt.xlabel("Accuracy")
    plt.ylabel("Algorithm")
    sns.barplot(x=scores, y=models, palette='dark')
    plt.show()

# -------------------------------
# Load Trained Model
# -------------------------------
def load_model(name="RandomForest"):
    with open(f"{name}.pkl", "rb") as f:
        model = pickle.load(f)
    return model

# -------------------------------
# Predict Crop
# -------------------------------
def predict_crop(model, features):
    features = np.array([features])
    return model.predict(features)[0]

# -------------------------------
# Train All and Return Accuracy
# -------------------------------
def train_and_evaluate_all():
    df = load_dataset()
    features = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    target = df['label']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=2)

    models = train_models(X_train, y_train)
    acc_dict = {}
    for name, model in models.items():
        acc, _ = evaluate_model(model, X_test, y_test)
        acc_dict[name] = acc
    return acc_dict
